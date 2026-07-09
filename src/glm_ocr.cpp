#include "glm_ocr.h"
#include "bpe_tokenizer.h"
#include "image_utils.h"
#include "rope_embed.h"
#include "vision_rope.h"
#include "sampling.h"
#include "text_runtime.h"

#include <fstream>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ============================================================
// GLMContext — GLM-OCR inference state
// ============================================================

struct GLMContext : public OCRContext {
    std::shared_ptr<OCRContext> clone() const override {
        auto dst = std::make_shared<GLMContext>();
        dst->kv_cache.resize(kv_cache.size());
        for (size_t i = 0; i < kv_cache.size(); ++i) {
            dst->kv_cache[i].first = kv_cache[i].first;
            dst->kv_cache[i].second = kv_cache[i].second;
        }
        dst->cur_token = cur_token;
        dst->position_id = position_id;
        return dst;
    }
};

// ============================================================
// Constructor
// ============================================================

GLMOCR::GLMOCR(const std::string& model_path, int num_threads)
    : num_threads_(num_threads > 0 ? num_threads : 4) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs.is_open()) {
                throw std::runtime_error("Cannot open model.json in " + model_path);
            }
            ifs >> config;
        }

        vision_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();

        if (num_threads_ > 0) {
            vision_net_->opt.num_threads = num_threads_;
            text_embed_net_->opt.num_threads = num_threads_;
            text_decoder_net_->opt.num_threads = num_threads_;
            lm_head_net_->opt.num_threads = num_threads_;
        }

        // Disable FP16/BF16 for all nets (use float32)
        for (auto* net : {vision_net_.get(), text_embed_net_.get(),
                          text_decoder_net_.get(), lm_head_net_.get()}) {
            net->opt.use_fp16_packed = false;
            net->opt.use_fp16_storage = false;
            net->opt.use_fp16_arithmetic = false;
            net->opt.use_bf16_storage = false;
        }

        // Load model files
        std::string vision_param = model_path + "/" + config["params"]["vision_param"].get<std::string>();
        std::string vision_bin = model_path + "/" + config["params"]["vision_bin"].get<std::string>();
        std::string text_embed_param = model_path + "/" + config["params"]["text_embed_param"].get<std::string>();
        std::string text_embed_bin = model_path + "/" + config["params"]["text_embed_bin"].get<std::string>();
        std::string text_decoder_param = model_path + "/" + config["params"]["text_decoder_param"].get<std::string>();
        std::string text_decoder_bin = model_path + "/" + config["params"]["text_decoder_bin"].get<std::string>();
        std::string lm_head_param = model_path + "/" + config["params"]["lm_head_param"].get<std::string>();
        std::string lm_head_bin = model_path + "/" + config["params"]["lm_head_bin"].get<std::string>();

        printf("[GLMOCR] Loading model from %s\n", model_path.c_str());

        vision_net_->load_param(vision_param.c_str());
        vision_net_->load_model(vision_bin.c_str());
        text_embed_net_->load_param(text_embed_param.c_str());
        text_embed_net_->load_model(text_embed_bin.c_str());
        text_decoder_net_->load_param(text_decoder_param.c_str());
        text_decoder_net_->load_model(text_decoder_bin.c_str());
        lm_head_net_->load_param(lm_head_param.c_str());
        lm_head_net_->load_model(lm_head_bin.c_str());

        // Load tokenizer
        std::string type = "bpe";
        if (config["tokenizer"].contains("type")) {
            type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        bpe_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"
        ));

        std::vector<std::string> additional_special_tokens =
            config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
        for (const auto& token : additional_special_tokens) {
            bpe_->AddAdditionalSpecialToken(token);
        }
        for (int id : bpe_->additional_special_token_ids()) {
            additional_special_id_set_.insert(id);
        }

        auto eos_token = config["tokenizer"]["eos"].get<std::string>();
        eos_ = (eos_token != "") ? bpe_->token_to_id().at(eos_token) : -1;
        auto it_eop = bpe_->token_to_id().find("<eop>");
        eop_ = (it_eop != bpe_->token_to_id().end()) ? it_eop->second : -1;

        // Full stop-token set. GLM ends the assistant turn with <|user|> (59253) as well
        // as <|endoftext|> (59246) — see generation_config.json eos_token_id.
        if (eos_ >= 0) eos_ids_.insert(eos_);
        if (eop_ >= 0) eos_ids_.insert(eop_);
        if (config["tokenizer"].contains("eos_ids")) {
            for (int id : config["tokenizer"]["eos_ids"].get<std::vector<int>>())
                eos_ids_.insert(id);
        }

        // Read model settings
        if (config["setting"].contains("attn_cnt")) {
            attn_cnt_ = config["setting"]["attn_cnt"].get<int>();
        }
        if (config["setting"].contains("hidden_size")) {
            hidden_size_ = config["setting"]["hidden_size"].get<int>();
        }
        if (config["setting"].contains("head_dim")) {
            head_dim_ = config["setting"]["head_dim"].get<int>();
        }
        if (config["setting"].contains("rope")) {
            auto text_rope_cfg = config["setting"]["rope"];
            if (!text_rope_cfg.contains("type") || text_rope_cfg["type"].get<std::string>() != "mRoPE") {
                throw std::runtime_error("unsupported setting.rope.type in model.json");
            }
            if (text_rope_cfg.contains("rope_theta")) {
                rope_theta_ = text_rope_cfg["rope_theta"].get<float>();
            }
            if (text_rope_cfg.contains("rope_head_dim")) {
                head_dim_ = text_rope_cfg["rope_head_dim"].get<int>();
            }
            if (!text_rope_cfg.contains("mrope_section")) {
                throw std::runtime_error("missing setting.rope.mrope_section in model.json");
            }
            for (auto& v : text_rope_cfg["mrope_section"]) {
                mrope_section_.push_back(v.get<int>());
            }
        } else {
            throw std::runtime_error("missing setting.rope in model.json");
        }
        if (mrope_section_.size() != 3) {
            throw std::runtime_error("setting.rope.mrope_section must have 3 entries");
        }
        if (config["setting"].contains("image_token_id")) {
            image_token_id_ = config["setting"]["image_token_id"].get<int>();
        }

        if (config["setting"].contains("vision")) {
            auto vision_cfg = config["setting"]["vision"];
            if (vision_cfg.contains("patch_size")) {
                patch_size_ = vision_cfg["patch_size"].get<int>();
            }
            if (vision_cfg.contains("spatial_merge_size")) {
                spatial_merge_size_ = vision_cfg["spatial_merge_size"].get<int>();
            }
            if (vision_cfg.contains("vision_hidden_size")) {
                vision_hidden_size_ = vision_cfg["vision_hidden_size"].get<int>();
            }
            if (vision_cfg.contains("vision_head_dim")) {
                vision_head_dim_ = vision_cfg["vision_head_dim"].get<int>();
            }
            if (vision_cfg.contains("vision_num_heads")) {
                vision_num_heads_ = vision_cfg["vision_num_heads"].get<int>();
            }
            if (!vision_cfg.contains("rope")) {
                throw std::runtime_error("missing setting.vision.rope in model.json");
            }
            auto vision_rope_cfg = vision_cfg["rope"];
            if (!vision_rope_cfg.contains("type") || vision_rope_cfg["type"].get<std::string>() != "mRoPE") {
                throw std::runtime_error("unsupported setting.vision.rope.type in model.json");
            }
            if (vision_rope_cfg.contains("rope_theta")) {
                vision_rope_theta_ = vision_rope_cfg["rope_theta"].get<float>();
            }
            if (vision_rope_cfg.contains("rope_head_dim")) {
                vision_rope_dim_ = vision_rope_cfg["rope_head_dim"].get<int>();
            }
            if (!vision_rope_cfg.contains("mrope_section")) {
                throw std::runtime_error("missing setting.vision.rope.mrope_section in model.json");
            }
            for (auto& v : vision_rope_cfg["mrope_section"]) {
                vision_mrope_section_.push_back(v.get<int>());
            }
            if (vision_rope_theta_ <= 0.0f || vision_rope_dim_ <= 0 || (vision_rope_dim_ % 2) != 0) {
                throw std::runtime_error("invalid setting.vision.rope rope_theta/rope_head_dim in model.json");
            }
            if (vision_mrope_section_.size() != 2 ||
                vision_mrope_section_[0] + vision_mrope_section_[1] != vision_rope_dim_) {
                throw std::runtime_error("setting.vision.rope.mrope_section must be [h_dim,w_dim] and sum to rope_head_dim");
            }
            if (vision_cfg.contains("max_num_patches")) {
                max_num_patches_ = vision_cfg["max_num_patches"].get<int>();
            }
            if (vision_cfg.contains("min_pixels")) {
                min_pixels_ = vision_cfg["min_pixels"].get<long long>();
            }
            if (vision_cfg.contains("max_pixels")) {
                max_pixels_ = vision_cfg["max_pixels"].get<long long>();
            }
            if (vision_cfg.contains("image_mean")) {
                auto mean = vision_cfg["image_mean"].get<std::vector<float>>();
                image_mean_[0] = mean[0]; image_mean_[1] = mean[1]; image_mean_[2] = mean[2];
            }
            if (vision_cfg.contains("image_std")) {
                auto std_vals = vision_cfg["image_std"].get<std::vector<float>>();
                image_std_[0] = std_vals[0]; image_std_[1] = std_vals[1]; image_std_[2] = std_vals[2];
            }
        }

        vocab_size_ = (int)bpe_->vocab_size();
        printf("  attn_cnt: %d, hidden_size: %d, head_dim: %d, vocab_size: %d\n",
               attn_cnt_, hidden_size_, head_dim_, vocab_size_);
        printf("  patch_size: %d, spatial_merge_size: %d, max_num_patches: %d\n",
               patch_size_, spatial_merge_size_, max_num_patches_);
        printf("  text mrope_section: [%d %d %d]\n",
               mrope_section_[0], mrope_section_[1], mrope_section_[2]);
        printf("  vision rope: type=mRoPE, rope_theta=%.1f, rope_head_dim=%d, mrope_section=[%d %d]\n",
               vision_rope_theta_, vision_rope_dim_, vision_mrope_section_[0], vision_mrope_section_[1]);
        printf("  image_token_id: %d, eos: %d, eop: %d\n", image_token_id_, eos_, eop_);

        ok_ = true;
    } catch (std::exception &e) {
        fprintf(stderr, "[GLMOCR] Load failed: %s\n", e.what());
        ok_ = false;
    }
}

GLMOCR::~GLMOCR() = default;

// ============================================================
// Image preprocessing
// ============================================================

void GLMOCR::get_image_size_for_patches(int img_h, int img_w, int& target_h, int& target_w) const {
    int effective_patch_size = patch_size_ * spatial_merge_size_;

    auto round_by_factor = [&](double size) -> int {
        int result = (int)(std::round(size / (double)effective_patch_size) * effective_patch_size);
        return std::max(effective_patch_size, result);
    };

    double h = (double)img_h;
    double w = (double)img_w;
    double area = h * w;

    double scale = 1.0;
    if (area > (double)max_pixels_) {
        scale = std::sqrt((double)max_pixels_ / area);
    } else if (area < (double)min_pixels_) {
        scale = std::sqrt((double)min_pixels_ / area);
    }

    target_h = round_by_factor(h * scale);
    target_w = round_by_factor(w * scale);
}

ncnn::Mat GLMOCR::bgr_to_image_strip(const ncnn::Mat& bgr, int& num_patches_h, int& num_patches_w) const {
    int img_h = bgr.h;
    int img_w = bgr.w;

    int target_h, target_w;
    get_image_size_for_patches(img_h, img_w, target_h, target_w);

    ncnn::Mat bgr_resized = ncnn_mat_resize(bgr, target_w, target_h);

    num_patches_h = target_h / patch_size_;
    num_patches_w = target_w / patch_size_;
    int num_patches = num_patches_h * num_patches_w;

    const unsigned char* bgr_data = (const unsigned char*)bgr_resized.data;

    // Match the TorchScript wrapper:
    // pv.view(1, N, 3, 2, 14, 14)[:, :, :, 0].permute(0, 2, 3, 1, 4)
    //   .reshape(1, 3, 14, 14 * N)
    ncnn::Mat image_strip(patch_size_ * num_patches, patch_size_, 3);
    image_strip.fill(0.0f);

    int grid_h = num_patches_h / spatial_merge_size_;
    int grid_w = num_patches_w / spatial_merge_size_;
    int patch_idx = 0;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int mh = 0; mh < spatial_merge_size_; mh++) {
                for (int mw = 0; mw < spatial_merge_size_; mw++) {
                    int patch_h = gh * spatial_merge_size_ + mh;
                    int patch_w = gw * spatial_merge_size_ + mw;
                    int start_y = patch_h * patch_size_;
                    int start_x = patch_w * patch_size_;
                    int strip_base_x = patch_idx * patch_size_;

                    for (int y = 0; y < patch_size_; y++) {
                        const unsigned char* img_row_ptr = bgr_data + (start_y + y) * target_w * 3;
                        float* dst_r = image_strip.channel(0).row(y) + strip_base_x;
                        float* dst_g = image_strip.channel(1).row(y) + strip_base_x;
                        float* dst_b = image_strip.channel(2).row(y) + strip_base_x;

                        for (int x = 0; x < patch_size_; x++) {
                            const unsigned char* pixel = img_row_ptr + (start_x + x) * 3;
                            dst_r[x] = (pixel[2] / 255.0f - image_mean_[0]) / image_std_[0];
                            dst_g[x] = (pixel[1] / 255.0f - image_mean_[1]) / image_std_[1];
                            dst_b[x] = (pixel[0] / 255.0f - image_mean_[2]) / image_std_[2];
                        }
                    }

                    patch_idx++;
                }
            }
        }
    }

    return image_strip;
}

// ============================================================
// Vision encoder
// ============================================================

ncnn::Mat GLMOCR::run_vision(const ncnn::Mat& image_strip, const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache) const {
    ncnn::Mat vision_features;
    ncnn::Extractor ex = vision_net_->create_extractor();
    ex.input("in0", image_strip);
    ex.input("in1", cos_cache);
    ex.input("in2", sin_cache);
    ex.extract("out0", vision_features);
    return vision_features;
}

// ============================================================
// Text RoPE cache
// ============================================================

void GLMOCR::generate_text_rope_cache(int seq_len, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const {
    // GLM-OCR text decoder uses interleaved RoPE (head_dim=128, but cos/sin are [seq_len, 64])
    generate_rope_embed_cache(seq_len, head_dim_, position_id, cos_cache, sin_cache, rope_theta_);
}

// ============================================================
// Prefill
// ============================================================

std::shared_ptr<OCRContext> GLMOCR::prefill(const std::string& prompt_text, const std::string& image_path) {
    // Step 1: Load image
    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) {
        fprintf(stderr, "[GLMOCR] Failed to load image: %s\n", image_path.c_str());
        return nullptr;
    }
    printf("[GLMOCR] Image: %dx%d\n", bgr.w, bgr.h);

    // Step 2: Image strip preprocessing
    int num_patches_h = 0, num_patches_w = 0;
    ncnn::Mat image_strip = bgr_to_image_strip(bgr, num_patches_h, num_patches_w);
    int num_patches = num_patches_h * num_patches_w;

    // Step 3: Generate vision RoPE cache and run vision encoder
    ncnn::Mat vision_cos, vision_sin;
    generate_vision_rope_cache_2d(num_patches_h, num_patches_w, spatial_merge_size_,
                                  vision_rope_theta_, vision_mrope_section_,
                                  false, vision_cos, vision_sin);

    ncnn::Mat vision_features = run_vision(image_strip, vision_cos, vision_sin);
    int num_vision_tokens = vision_features.h;
    printf("[GLMOCR] Vision tokens: %d\n", num_vision_tokens);

    // Step 4: Build GLM-OCR chat template
    // [gMASK]<sop><|user|>\n<|begin_of_image|><|image|>×N<|end_of_image|>{prompt}/nothink<|assistant|>\n thinking
    std::string full_prompt = "[gMASK]<sop><|user|>\n<|begin_of_image|>";
    for (int i = 0; i < num_vision_tokens; i++) {
        full_prompt += "<|image|>";
    }
    full_prompt += "<|end_of_image|>" + prompt_text;
    if (prompt_text.size() < 8 || prompt_text.rfind("/nothink") != prompt_text.size() - 8) {
        full_prompt += "/nothink";
    }
    full_prompt += "<|assistant|>\n thinking";

    // Step 5: Tokenize
    std::vector<int> token_ids = bpe_->encode(full_prompt, false, false);
    printf("[GLMOCR] Prompt: %zu tokens\n", token_ids.size());

    // Step 6: Get text embeddings
    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids);

    // Step 7: Find image token positions and inject vision features
    std::vector<int> image_token_positions;
    for (int i = 0; i < (int)token_ids.size(); i++) {
        if (token_ids[i] == image_token_id_) {
            image_token_positions.push_back(i);
        }
    }
    token_embed = token_embed.clone();
    if ((int)image_token_positions.size() == num_vision_tokens) {
        for (int i = 0; i < num_vision_tokens; i++) {
            int pos = image_token_positions[i];
            float* embed_ptr = token_embed.row(pos);
            const float* feat_ptr = vision_features.row(i);
            memcpy(embed_ptr, feat_ptr, hidden_size_ * sizeof(float));
        }
    } else {
        printf("[GLMOCR] WARNING: image token count mismatch! vision_tokens=%d, image_tokens_in_prompt=%d\n",
               num_vision_tokens, (int)image_token_positions.size());
    }

    // Step 8: Generate causal mask
    int seq_len = (int)token_ids.size();
    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) {
            row[j] = -1e38f;
        }
    }

    // Step 9: Generate MRoPE cache for text decoder
    ncnn::Mat cos_cache, sin_cache;
    int next_position_id = seq_len;
    if (!mrope_section_.empty() && !image_token_positions.empty()) {
        generate_rope_embed_cache_vision_mrope(seq_len, head_dim_, 0,
                                               image_token_positions[0], num_vision_tokens,
                                               num_patches_w, spatial_merge_size_,
                                               mrope_section_, cos_cache, sin_cache, rope_theta_);
        next_position_id = seq_len - num_vision_tokens + (num_patches_w / spatial_merge_size_);
    } else {
        generate_text_rope_cache(seq_len, 0, cos_cache, sin_cache);
    }

    // Step 10: Run decoder (prefill pass)
    KVCache kv_cache;
    ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask,
                                                     cos_cache, sin_cache,
                                                     kv_cache, attn_cnt_, true);

    // Step 11: Run LM head on the last token
    ncnn::Mat last_hidden = decode_out.row_range(seq_len - 1, 1);
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);

    // Step 12: Get next token via argmax
    int next_token_id = argmax1d(logits);

    printf("[GLMOCR] Prefill done, next_token=%d\n", next_token_id);

    // Step 13: Create context
    auto ctx = std::make_shared<GLMContext>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = next_position_id;
    return ctx;
}

// ============================================================
// Generate loop
// ============================================================

void GLMOCR::generate(std::shared_ptr<OCRContext> ctx_base,
                      const GenerateConfig& cfg,
                      std::function<void(const std::string&)> callback) {
    auto ctx = std::static_pointer_cast<GLMContext>(ctx_base);

    std::unordered_set<int> history;
    history.insert(ctx->cur_token);
    std::string emitted_text;

    LlmTokenSampleConfig sample_cfg;
    sample_cfg.vocab_size = vocab_size_;
    sample_cfg.temperature = cfg.temperature;
    sample_cfg.top_p = cfg.top_p;
    sample_cfg.top_k = cfg.top_k;
    sample_cfg.repetition_penalty = cfg.repetition_penalty;
    sample_cfg.do_sample = cfg.do_sample;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (eos_ids_.count(ctx->cur_token)) break;

        // Skip outputting special tokens (e.g. </think|>)
        bool is_special = eos_ids_.count(ctx->cur_token) ||
                          (additional_special_id_set_.find(ctx->cur_token) != additional_special_id_set_.end());
        if (!is_special) {
            std::string token_text = bpe_->decode({ctx->cur_token}, false);

            // Stop if ``` appears after first token
            if (!emitted_text.empty() && token_text.find("```") != std::string::npos) {
                break;
            }

            // Stop if 3+ trailing newlines
            std::string candidate_text = emitted_text + token_text;
            int trailing_newlines = 0;
            for (auto it = candidate_text.rbegin(); it != candidate_text.rend(); ++it) {
                if (*it == '\n') {
                    trailing_newlines++;
                } else if (*it == '\r') {
                    continue;
                } else {
                    break;
                }
            }
            if (!emitted_text.empty() && trailing_newlines > 2) {
                break;
            }
            emitted_text += token_text;
            callback(token_text);
        }

        // Single-token embedding
        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, ctx->cur_token);

        // Generate RoPE cache for single token
        ncnn::Mat cos_cache, sin_cache;
        generate_text_rope_cache(1, ctx->position_id, cos_cache, sin_cache);
        ctx->position_id++;

        // Mask: [1, kv_len+1] all zeros
        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
        mask.fill(0.0f);

        // Decoder with KV cache (incremental)
        ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, cur_embed,
                                                         mask, cos_cache, sin_cache,
                                                         ctx->kv_cache, attn_cnt_, false);

        // LM head
        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);

        // Sample next token
        int next_id = llm_select_next_token(logits, history, sample_cfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }
}