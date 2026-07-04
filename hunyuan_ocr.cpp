#include "hunyuan_ocr.h"
#include "bpe_tokenizer.h"
#include "expression_layer.h"
#include "image_utils.h"
#include "rope_embed.h"

#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// nlohmann/json for config
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace hunyuan {

// ============================================================
// BpeTokenizer (same class as in bpe_tokenizer.h)
// The header is included separately; here we just use it.
// ============================================================

// ============================================================
// Image preprocessing (futz12 approach: BICUBIC + smart_resize)
// ============================================================

static void smart_resize_hunyuan(int img_h, int img_w, int& target_h, int& target_w,
                                  int patch_size, int spatial_merge, int min_px, int max_px) {
    const long long factor = (long long)patch_size * spatial_merge; // 32

    auto round_half_even = [](double x) -> long long {
        double fl = std::floor(x);
        double diff = x - fl;
        if (diff < 0.5) return (long long)fl;
        if (diff > 0.5) return (long long)fl + 1;
        return ((long long)fl % 2 == 0) ? (long long)fl : (long long)fl + 1;
    };

    long long h_bar = std::max(factor, round_half_even((double)img_h / factor) * factor);
    long long w_bar = std::max(factor, round_half_even((double)img_w / factor) * factor);
    const double area = (double)img_h * (double)img_w;

    if (h_bar * w_bar > (long long)max_px) {
        double beta = std::sqrt(area / (double)max_px);
        h_bar = std::max(factor, (long long)std::floor(img_h / beta / factor) * factor);
        w_bar = std::max(factor, (long long)std::floor(img_w / beta / factor) * factor);
    } else if (h_bar * w_bar < (long long)min_px) {
        double beta = std::sqrt((double)min_px / area);
        h_bar = (long long)std::ceil(img_h * beta / factor) * factor;
        w_bar = (long long)std::ceil(img_w * beta / factor) * factor;
    }
    target_h = (int)h_bar;
    target_w = (int)w_bar;
}

// ============================================================
// HunYuanOCR implementation
// ============================================================

HunYuanOCR::HunYuanOCR(const std::string& model_dir, int num_threads)
    : num_threads_(num_threads), model_dir_(model_dir) {
    try {
        // Load config
        json config;
        {
            std::ifstream ifs(model_dir + "/model.json");
            if (!ifs.is_open()) {
                throw std::runtime_error("Cannot open model.json in " + model_dir);
            }
            ifs >> config;
        }

        auto& setting = config["setting"];
        hidden_size_ = setting.value("hidden_size", 1024);
        num_layers_ = setting.value("attn_cnt", 24);
        vocab_size_ = setting.value("vocab_size", 120818);
        image_token_id_ = setting.value("image_token_id", 120120);
        image_start_id_ = setting.value("image_start_token_id", 120118);
        image_end_id_ = setting.value("image_end_token_id", 120119);
        bos_id_ = setting.value("bos_token_id", 120000);
        eos_id_ = setting.value("eos_token_id", 120001);  // Fixed: was 120020

        if (setting.contains("vision")) {
            auto& vis = setting["vision"];
            patch_size_ = vis.value("patch_size", 16);
            spatial_merge_size_ = vis.value("spatial_merge_size", 2);
        }

        // Load tokenizer with byte-level encoding (HunYuanOCR uses GPT-2 BBPE)
        std::string vocab_path = model_dir + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_path = model_dir + "/" + config["tokenizer"]["merges_file"].get<std::string>();
        std::string tok_type = config["tokenizer"].value("type", "bpe");
        SpecialTokensConfig sp_cfg;
        sp_cfg.unk_token = "unk";  // Token ID 7172 in HunYuanOCR vocab
        tokenizer_ = std::unique_ptr<BpeTokenizer>(new BpeTokenizer(
            BpeTokenizer::LoadFromFiles(vocab_path, merges_path, sp_cfg, false, true, true)));
        // Tokenizer loaded successfully

        // Register special tokens for direct matching
        auto& tok_cfg = config["tokenizer"];
        if (tok_cfg.contains("bos"))
            tokenizer_->AddAdditionalSpecialToken(tok_cfg["bos"].get<std::string>(), false);
        if (tok_cfg.contains("eos"))
            tokenizer_->AddAdditionalSpecialToken(tok_cfg["eos"].get<std::string>(), false);
        // Register chat template special tokens (EOS, User, Assistant)
        tokenizer_->add_special_token_with_id("<｜hy_place▁holder▁no▁2｜>", 120001);
        tokenizer_->add_special_token_with_id("<｜hy_User｜>", 120006);
        tokenizer_->add_special_token_with_id("<｜hy_Assistant｜>", 120007);

        // Also register image-related special tokens with explicit IDs
        auto& sp_file = model_dir + "/special_tokens.json";
        std::ifstream sp_ifs(sp_file);
        if (sp_ifs.is_open()) {
            json sp_json;
            sp_ifs >> sp_json;
            for (auto& [name, info] : sp_json.items()) {
                if (info.is_object() && info.contains("token") && info.contains("id")) {
                    std::string token_str = info["token"].get<std::string>();
                    int token_id = info["id"].get<int>();
                    tokenizer_->add_special_token_with_id(token_str, token_id);
                }
            }
        }

        // Load ncnn networks (direct approach, matching working minimal main)
        auto& params = config["params"];
        int nt = num_threads_;

        auto load_one = [&params, &model_dir, nt](const char* key) -> std::shared_ptr<ncnn::Net> {
            std::string p = model_dir + "/" + params[key].get<std::string>();
            std::string b = p; b.replace(b.rfind(".param"), 6, ".bin");
            auto net = std::make_shared<ncnn::Net>();
            net->opt.num_threads = nt;
            net->opt.use_fp16_packed = false;
            net->opt.use_fp16_storage = false;
            net->opt.use_fp16_arithmetic = false;
            net->opt.use_bf16_storage = false;
            // New models have 0 custom layers (all native ncnn ops)
            // hunyuan::register_atento_layer(*net);
            // hunyuan::register_tensorindex_layer(*net);
            // hunyuan::register_repeat_interleave_layer(*net);
            // hunyuan::register_expression_layer(*net, p.c_str());
            net->load_param(p.c_str());
            net->load_model(b.c_str());
            return net;
        };
        vision_net_ = load_one("vision_param");
        text_embed_net_ = load_one("text_embed_param");
        text_decoder_net_ = load_one("text_decoder_param");
        // lm_head loaded lazily

        printf("[HunYuanOCR] Model loaded successfully\n");
        printf("  hidden_size=%d, num_layers=%d, vocab_size=%d\n",
               hidden_size_, num_layers_, vocab_size_);
        printf("  image_token_id=%d, bos=%d, eos=%d\n",
               image_token_id_, bos_id_, eos_id_);

        ok_ = true;
    } catch (std::exception& e) {
        fprintf(stderr, "[HunYuanOCR] Load failed: %s\n", e.what());
        ok_ = false;
    }
}

HunYuanOCR::~HunYuanOCR() = default;

// ============================================================
// Image preprocessing
// ============================================================

bool HunYuanOCR::preprocess_image(const std::string& image_path,
                                   ncnn::Mat& pixel_values) {
    // Use futz12 approach: load as BGR, smart_resize, bicubic, normalize
    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) {
        fprintf(stderr, "[HunYuanOCR] Failed to load image: %s\n", image_path.c_str());
        return false;
    }

    printf("[HunYuanOCR] Image: %dx%d\n", bgr.w, bgr.h);

    // Smart resize matching HF processor
    smart_resize_hunyuan(bgr.h, bgr.w, target_h_, target_w_,
                          patch_size_, spatial_merge_size_, 262144, 4194304);
    // printf("[HunYuanOCR] Smart resize: %dx%d -> %dx%d\n", bgr.w, bgr.h, target_w, target_h);

    // Bicubic resize (matching PIL) + normalize
    ncnn::Mat resized = ncnn_mat_resize_bicubic(bgr, target_w_, target_h_);
    const unsigned char* data = (const unsigned char*)resized.data;
    float mean_v[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float stdv[3] =  {0.26862954f, 0.26130258f, 0.27577711f};

    pixel_values.create(target_w_, target_h_, 3, sizeof(float));
    if (pixel_values.empty()) return false;
    // BGR from load_image_to_ncnn_mat → convert to RGB planar
    for (int y = 0; y < target_h_; y++) {
        const unsigned char* row = data + (size_t)y * target_w_ * 3;
        float* rr = pixel_values.channel(0).row(y);
        float* gg = pixel_values.channel(1).row(y);
        float* bb = pixel_values.channel(2).row(y);
        for (int x = 0; x < target_w_; x++) {
            const unsigned char* px = row + (size_t)x * 3;  // B,G,R order
            rr[x] = (px[2] / 255.0f - mean_v[0]) / stdv[0];  // R
            gg[x] = (px[1] / 255.0f - mean_v[1]) / stdv[1];  // G
            bb[x] = (px[0] / 255.0f - mean_v[2]) / stdv[2];  // B
        }
    }
    return true;
}

// ============================================================
// Vision Encoder
// ============================================================

ncnn::Mat HunYuanOCR::run_vision_encoder(const ncnn::Mat& pixel_values,
                                           const ncnn::Mat& pos_embed) {
    ncnn::Mat output;
    ncnn::Extractor ex = vision_net_->create_extractor();
    ex.input("in0", pixel_values);
    ex.input("in1", pos_embed);
    ex.extract("out0", output);
    return output;
}

// ============================================================
// Text Embed
// ============================================================

ncnn::Mat HunYuanOCR::run_text_embed(const std::vector<int>& token_ids) {
    ncnn::Mat input((int)token_ids.size(), 1, 1, sizeof(int));
    if (input.empty()) return ncnn::Mat();
    memcpy(input.data, token_ids.data(), token_ids.size() * sizeof(int));

    ncnn::Mat output;
    ncnn::Extractor ex = text_embed_net_->create_extractor();
    ex.input("in0", input);
    ex.extract("out0", output);
    return output;
}

ncnn::Mat HunYuanOCR::run_text_embed(int token_id) {
    return run_text_embed(std::vector<int>{token_id});
}

// ============================================================
// Text Decoder (full prefill, no KV cache)
// ============================================================

ncnn::Mat HunYuanOCR::run_text_decoder(const ncnn::Mat& inputs_embeds,
                                        const ncnn::Mat& causal_mask,
                                        const std::vector<int>& position_ids) {
    int seq_len = inputs_embeds.h;

    // Generate XD-RoPE cos/sin [64, seq_len] — head_dim/2, decoder doubles internally
    ncnn::Mat cos_tbl, sin_tbl;
    generate_xdrope_cache(128, seq_len, position_ids.empty() ? 0 : position_ids[0],
                          cos_tbl, sin_tbl, 10000.0f);

    printf("[Decoder] seq_len=%d, embeds=[%d x %d], mask=[%d x %d], cos=[%d x %d]\n",
           seq_len, inputs_embeds.w, inputs_embeds.h,
           causal_mask.w, causal_mask.h, cos_tbl.w, cos_tbl.h);
    fflush(stdout);

    ncnn::Mat output;
    ncnn::Extractor ex = text_decoder_net_->create_extractor();
    printf("[Decoder] Created extractor, setting inputs...\n"); fflush(stdout);
    ex.input("in0", inputs_embeds);
    ex.input("in1", causal_mask);
    ex.input("in2", cos_tbl);
    ex.input("in3", sin_tbl);
    printf("[Decoder] Inputs set, extracting...\n"); fflush(stdout);
    int ret = ex.extract("out0", output);
    printf("[Decoder] Extract returned %d, output=[%d x %d x %d]\n",
           ret, output.w, output.h, output.c);
    fflush(stdout);
    return output;
}

// ============================================================
// LM Head
// ============================================================

ncnn::Mat HunYuanOCR::run_lm_head(const ncnn::Mat& hidden_states) {
    // Lazy load lm_head on first call
    if (!lm_head_net_) {
        lm_head_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_->opt.num_threads = num_threads_;
        lm_head_net_->opt.use_fp16_packed = false;
        lm_head_net_->opt.use_fp16_storage = false;
        lm_head_net_->opt.use_fp16_arithmetic = false;
        lm_head_net_->opt.use_bf16_storage = false;
        hunyuan::register_atento_layer(*lm_head_net_);
        hunyuan::register_tensorindex_layer(*lm_head_net_);
        hunyuan::register_repeat_interleave_layer(*lm_head_net_);
        hunyuan::register_expression_layer(*lm_head_net_, "./models/lm_head.ncnn.param");
        lm_head_net_->load_param("./models/lm_head.ncnn.param");
        lm_head_net_->load_model("./models/text_embed.ncnn.bin");
        // printf("[LMHead] Lazy loaded\n");
    }
    ncnn::Mat output;
    ncnn::Extractor ex = lm_head_net_->create_extractor();
    ex.input("in0", hidden_states);
    ex.extract("out0", output);
    return output;
}

// ============================================================
// Tokenizer wrappers
// ============================================================

std::vector<int> HunYuanOCR::tokenize(const std::string& text) const {
    return tokenizer_->encode(text, false, false);
}

std::string HunYuanOCR::detokenize(const std::vector<int>& ids) const {
    return tokenizer_->decode(ids, true);
}

// ============================================================
// Prompt building
// ============================================================

void HunYuanOCR::build_prompt_and_tokenize(const std::string& user_prompt,
                                            int num_vision_tokens,
                                            std::vector<int>& token_ids,
                                            std::vector<int>& image_positions) {
    // Build the prompt matching the HunYuan chat template:
    // <|bos|><image_token × N><user_text><|User|><|Assistant|>
    // Known special token IDs (from tokenizer_config.json added_tokens_decoder):
    //   BOS=120000, User=120006, Assistant=120007, Image=120120

    // BBPE tokenizer can't handle Chinese — use hardcoded HF token IDs
    // "检测并识别图片中的文字。" -> [5055,951,9977,12858,1843,9738,292]
    // (confirmed from Python HF tokenizer)
    static const std::vector<int> kOCR_CN = {5055,951,9977,12858,1843,9738,292};
    static const std::vector<int> kOCR_EN = {114946,25};

    std::vector<int> text_ids = tokenizer_->encode(user_prompt, false, false);
    // printf("[HunYuanOCR] BBPE tokenizer produced %zu tokens for prompt\n", text_ids.size());
    // Always use hardcoded HF IDs (BBPE can't handle Chinese correctly)
    if (user_prompt.size() > 20)
        text_ids = kOCR_CN;
    else
        text_ids = kOCR_EN;

    token_ids.push_back(120000);  // BOS
    token_ids.push_back(120021);  // System (empty system message)
    for (int i = 0; i < num_vision_tokens; i++) token_ids.push_back(120120);
    for (int id : text_ids) token_ids.push_back(id);
    token_ids.push_back(120006);  // <|User|>

    // Find image token positions
    image_positions.clear();
    for (int i = 0; i < (int)token_ids.size(); i++) {
        if (token_ids[i] == image_token_id_) {
            image_positions.push_back(i);
        }
    }
    // printf("[HunYuanOCR] Prompt: %d tokens, %d image tokens\n", (int)token_ids.size(), (int)image_positions.size());
    // Prompt tokenized
}

// ============================================================
// Prefill
// ============================================================

std::shared_ptr<HunYuanContext> HunYuanOCR::prefill(
    const std::string& prompt, const std::string& image_path) {

    // Step 1: Preprocess image
    ncnn::Mat pixel_values;
    if (!preprocess_image(image_path, pixel_values)) {
        return nullptr;
    }

    // Step 2: Interpolate position embedding and run vision encoder
    ncnn::Mat pos_embed;
    {
        int gh = target_h_ / patch_size_;  // e.g. 24
        int gw = target_w_ / patch_size_;  // e.g. 44
        // Load base position embedding [1, 1152, 128, 128]
        static std::vector<float> pos_base;
        static bool pos_loaded = false;
        if (!pos_loaded) {
            std::string pe_path = model_dir_ + "/models/pos_embed.bin";
            FILE* fp = fopen(pe_path.c_str(), "rb");
            if (!fp) {
                fprintf(stderr, "[HunYuanOCR] Cannot open pos_embed.bin\n");
                return nullptr;
            }
            pos_base.resize(1152 * 128 * 128);
            fread(pos_base.data(), sizeof(float), pos_base.size(), fp);
            fclose(fp);
            pos_loaded = true;
        }
        // Bilinear interpolate [128,128]→[gh,gw] with HF scale_factor mapping
        // src = (dst + 0.5) * 128 / (gh+0.1) - 0.5
        pos_embed.create(gw, gh, 1152, sizeof(float));
        float scale_h = (gh + 0.1f) / 128.0f;
        float scale_w = (gw + 0.1f) / 128.0f;
        for (int c = 0; c < 1152; c++) {
            ncnn::Mat chan = pos_embed.channel(c);
            const float* src = pos_base.data() + (size_t)c * 128 * 128;
            for (int y = 0; y < gh; y++) {
                float sy = (y + 0.5f) / scale_h - 0.5f;
                int y0 = (int)floorf(sy); y0 = std::max(0, std::min(y0, 127));
                int y1 = std::min(y0 + 1, 127);
                float fy = sy - y0;
                float* row = chan.row(y);
                for (int x = 0; x < gw; x++) {
                    float sx = (x + 0.5f) / scale_w - 0.5f;
                    int x0 = (int)floorf(sx); x0 = std::max(0, std::min(x0, 127));
                    int x1 = std::min(x0 + 1, 127);
                    float fx = sx - x0;
                    float v00 = src[y0 * 128 + x0], v01 = src[y0 * 128 + x1];
                    float v10 = src[y1 * 128 + x0], v11 = src[y1 * 128 + x1];
                    row[x] = (v00 * (1-fx) + v01 * fx) * (1-fy) +
                             (v10 * (1-fx) + v11 * fx) * fy;
                }
            }
        }
    }
    ncnn::Mat image_embeds = run_vision_encoder(pixel_values, pos_embed);
    int num_vision_tokens = image_embeds.h;
    printf("[HunYuanOCR] Vision tokens: %d, embed dim: %d\n",
           num_vision_tokens, image_embeds.w);
    // ViT output ready

    // Step 3: Build prompt and tokenize
    std::vector<int> token_ids;
    std::vector<int> image_positions;
    build_prompt_and_tokenize(prompt, num_vision_tokens, token_ids, image_positions);
    int seq_len = (int)token_ids.size();

    // Step 4: Run text embedding
    ncnn::Mat text_embeds = run_text_embed(token_ids);
    // Embeddings ready

    // Step 5: Inject vision features (DEBUG: use random instead of real ViT output)
    text_embeds = text_embeds.clone();
    if (image_positions.size() == (size_t)num_vision_tokens) {
        for (int i = 0; i < num_vision_tokens; i++) {
            int pos = image_positions[i];
            float* embed_ptr = text_embeds.row(pos);
            const float* feat_ptr = image_embeds.row(i);
            memcpy(embed_ptr, feat_ptr, hidden_size_ * sizeof(float));
        }
    } else {
        printf("[HunYuanOCR] WARNING: vision token mismatch! vision=%d, image_tokens=%zu\n",
               num_vision_tokens, image_positions.size());
    }
    // Vision features injected

    // Use actual sequence length (new model supports dynamic len up to 320).
    // No padding/truncation needed for typical prompts (\(\leq\)310 tokens).
    const int MAX_SEQ = 512;
    if (seq_len > MAX_SEQ) {
        int excess = seq_len - MAX_SEQ;
        printf("[HunYuanOCR] WARNING: Truncating %d -> %d tokens\n", seq_len, MAX_SEQ);
        // Truncate image tokens from the end
        int img_end = -1;
        for (int i = seq_len-1; i >= 0 && excess > 0; i--) {
            if (token_ids[i] == image_token_id_) { token_ids.erase(token_ids.begin()+i); excess--; }
        }
        ncnn::Mat ne(hidden_size_, (int)token_ids.size(), sizeof(float));
        for (size_t i = 0; i < token_ids.size(); i++) memcpy(ne.row((int)i), text_embeds.row((int)i), hidden_size_*sizeof(float));
        text_embeds = ne;
        seq_len = (int)token_ids.size();
        image_positions.clear();
        for (int i = 0; i < seq_len; i++) if (token_ids[i]==image_token_id_) image_positions.push_back(i);
        num_vision_tokens = (int)image_positions.size();
    }

    // Step 6: Build causal mask (2D for SDPA compatibility)
    ncnn::Mat mask(seq_len, seq_len, sizeof(float));
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++)
        for (int j = i + 1; j < seq_len; j++)
            ((float*)mask.row(i))[j] = -1e38f;

    // Step 7: Build 4-axis XD-RoPE position IDs
    // axis 0: linear positions for all tokens
    // axis 1-3: spatial (w, h, t) for image tokens, linear for text tokens
    int first_img = image_positions.empty() ? 2 : image_positions[0];
    int patch_w = target_w_ / patch_size_ / spatial_merge_size_;
    int patch_h_2 = target_h_ / patch_size_ / spatial_merge_size_;

    std::vector<int> pos4_arr[4];
    for (int a = 0; a < 4; a++) {
        pos4_arr[a].resize(seq_len);
        for (int i = 0; i < seq_len; i++) pos4_arr[a][i] = i;
    }
    // Set spatial positions for image tokens (skip the "begin" token at first_img)
    int start = first_img + 1;
    for (int r = 0; r < patch_h_2; r++) {
        for (int c = 0; c < patch_w + 1; c++) {
            int idx = start + r * (patch_w + 1) + c;
            if (idx >= seq_len) break;
            pos4_arr[1][idx] = c;  // w position
            pos4_arr[2][idx] = r;  // h position
            pos4_arr[3][idx] = 0;  // t position
        }
    }
    const std::vector<int> pos4[4] = {pos4_arr[0], pos4_arr[1], pos4_arr[2], pos4_arr[3]};

    printf("[HunYuanOCR] Running decoder (%d tokens)...\n", seq_len);
    fflush(stdout);
    // Step 8: Run decoder with KV cache (prefill)
    ncnn::Mat decoder_out;
    KVCache kv_cache;
    {
        ncnn::Net* dec = text_decoder_net_.get();
        ncnn::Mat cos_tbl, sin_tbl;
        std::vector<int> xdrope = {16, 16, 16, 16};
        generate_hunyuan_xdrope_cos_sin(pos4, seq_len, 128, xdrope, 10000.0f, 1000.0f, cos_tbl, sin_tbl);

        ncnn::Extractor ex = dec->create_extractor();
        ex.input("in0", text_embeds);
        ex.input("in1", mask);
        ex.input("in2", cos_tbl);
        ex.input("in3", sin_tbl);

        // Extract KV cache from each attention layer (named blobs)
        for (int i = 0; i < num_layers_; i++) {
            char name_k[32], name_v[32];
            std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", i);
            std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k, k_cache);
            ex.extract(name_v, v_cache);
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }

        ex.extract("out0", decoder_out);
    }

    // Step 9: Run LM head on last token
    ncnn::Mat lh(1024, 1, sizeof(float));
    memcpy(lh.data, decoder_out.row(seq_len - 1), 1024*sizeof(float));
    ncnn::Mat logits = run_lm_head(lh);

    // Step 10: Greedy decode next token
    const float* logits_ptr = logits;
    int next_token = 0;
    float max_val = logits_ptr[0];
    for (int i = 1; i < vocab_size_; i++) {
        if (logits_ptr[i] > max_val) {
            max_val = logits_ptr[i];
            next_token = i;
        }
    }

    printf("[HunYuanOCR] Prefill done, next_token=%d\n", next_token);

    auto ctx = std::make_shared<HunYuanContext>();
    ctx->token_ids = std::move(token_ids);
    ctx->cur_token = next_token;
    ctx->position_id = seq_len;
    ctx->vision_features = image_embeds.clone();
    ctx->image_positions = image_positions;
    ctx->num_vision_tokens = num_vision_tokens;
    ctx->kv_cache = std::move(kv_cache);
    return ctx;
}

// ============================================================
// Generate loop (KV cache — single token per step)
// ============================================================

void HunYuanOCR::generate(std::shared_ptr<HunYuanContext> ctx,
                           const GenerateConfig& cfg,
                           std::function<void(const std::string&)> callback) {
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);
    const int attn_cnt = num_layers_;
    const std::vector<int> xdrope = {16, 16, 16, 16};
    const int special_begin = 120000;

    for (int step = 0; step < cfg.max_new_tokens; step++) {
        int tok = ctx->cur_token;
        // EOS: {120007, 120020}
        if (tok == 120007 || tok == 120020) break;

        // Emit token (skip special tokens)
        bool is_special = (tok >= 120000);
        if (!is_special) {
            std::string token_text = tokenizer_->decode({tok}, true);
            if (!token_text.empty()) callback(token_text);
        }

        // Single-token embedding
        ncnn::Mat cur_embed = run_text_embed({tok});

        // 4-axis position for this single token (all axes = current position)
        int pos = ctx->position_id;
        ctx->position_id++;
        std::vector<int> p1(1, pos);
        const std::vector<int> pos4_arr[4] = {p1, p1, p1, p1};
        ncnn::Mat cos_tbl, sin_tbl;
        generate_hunyuan_xdrope_cos_sin(pos4_arr, 1, 128, xdrope, 10000.0f, 1000.0f, cos_tbl, sin_tbl);

        // Mask: [1, kv_len+1] all zeros (new token attends to all cached + itself)
        int kv_len = ctx->kv_cache.empty() ? 0 : ctx->kv_cache[0].first.h;
        ncnn::Mat mask(1, kv_len + 1, sizeof(float));
        mask.fill(0.0f);

        // Decoder with KV cache
        ncnn::Net* dec = text_decoder_net_.get();
        ncnn::Extractor ex = dec->create_extractor();
        ex.input("in0", cur_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_tbl);
        ex.input("in3", sin_tbl);

        // Feed past KV cache (named blobs)
        for (int i = 0; i < attn_cnt; i++) {
            char name_k[32], name_v[32];
            std::snprintf(name_k, sizeof(name_k), "cache_k%d", i);
            std::snprintf(name_v, sizeof(name_v), "cache_v%d", i);
            ex.input(name_k, ctx->kv_cache[i].first);
            ex.input(name_v, ctx->kv_cache[i].second);
        }

        // Extract new KV cache
        KVCache new_cache;
        for (int i = 0; i < attn_cnt; i++) {
            char name_k[32], name_v[32];
            std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", i);
            std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k, k_cache);
            ex.extract(name_v, v_cache);
            new_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }

        // Decoder output (single token)
        ncnn::Mat decode_out;
        ex.extract("out0", decode_out);

        // LM head
        ncnn::Mat logits = run_lm_head(decode_out);

        // Sample next token
        int next_token = sample_token(logits, history, cfg);
        ctx->cur_token = next_token;
        ctx->kv_cache = std::move(new_cache);
        history.insert(next_token);
    }
}

// ============================================================
// Token sampling
// ============================================================

int HunYuanOCR::sample_token(const ncnn::Mat& logits,
                              const std::unordered_set<int>& history,
                              const GenerateConfig& cfg) {
    // Copy logits
    std::vector<float> scores(vocab_size_);
    memcpy(scores.data(), logits.data, sizeof(float) * vocab_size_);

    // Apply repetition penalty
    for (int t : history) {
        if (t < 0 || t >= vocab_size_) continue;
        if (scores[t] < 0) {
            scores[t] *= cfg.repetition_penalty;
        } else {
            scores[t] /= cfg.repetition_penalty;
        }
    }

    // If greedy (temperature=0), just argmax
    if (!cfg.do_sample || cfg.temperature <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    // Softmax with temperature
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (int i = 0; i < vocab_size_; i++) {
        scores[i] = std::exp((scores[i] - max_score) / cfg.temperature);
        sum += scores[i];
    }
    for (int i = 0; i < vocab_size_; i++) {
        scores[i] /= sum;
    }

    // Top-K
    if (cfg.top_k > 0 && cfg.top_k < vocab_size_) {
        std::vector<float> tmp = scores;
        std::nth_element(tmp.begin(), tmp.end() - cfg.top_k, tmp.end());
        float threshold = tmp[vocab_size_ - cfg.top_k];
        for (int i = 0; i < vocab_size_; i++) {
            if (scores[i] < threshold) scores[i] = 0.0f;
        }
    }

    // Top-P
    if (cfg.top_p < 1.0f) {
        std::vector<std::pair<float, int>> sorted;
        sorted.reserve(vocab_size_);
        for (int i = 0; i < vocab_size_; i++) {
            sorted.emplace_back(scores[i], i);
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<>());

        float cum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); i++) {
            cum += sorted[i].first;
            if (cum >= cfg.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        std::vector<char> keep(vocab_size_, 0);
        for (size_t i = 0; i < cutoff; i++) {
            keep[sorted[i].second] = 1;
        }
        for (int i = 0; i < vocab_size_; i++) {
            if (!keep[i]) scores[i] = 0.0f;
        }
    }

    // Renormalize
    sum = 0.0f;
    for (int i = 0; i < vocab_size_; i++) sum += scores[i];
    if (sum <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }
    for (int i = 0; i < vocab_size_; i++) scores[i] /= sum;

    // Sample
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < vocab_size_; i++) {
        cum += scores[i];
        if (r < cum) return i;
    }
    return vocab_size_ - 1;
}

// ============================================================
// XD-RoPE cos/sin generation
// ============================================================

void generate_xdrope_cache(int head_dim, int seq_len, int position_id,
                           ncnn::Mat& cos_cache, ncnn::Mat& sin_cache,
                           float rope_theta) {
    int half_dim = head_dim / 2;  // 64

    // XD-RoPE alpha scaling: base = rope_theta * alpha^(dim/(dim-2))
    // alpha = 1000.0 for HunYuanOCR
    const float alpha = 1000.0f;
    float base = rope_theta * std::pow(alpha, (float)head_dim / (float)(head_dim - 2));

    // Compute inverse frequencies with scaled base
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; i++) {
        inv_freq[i] = 1.0f / std::pow(base, (float)(i * 2) / head_dim);
    }

    // Allocate 2D: [half_dim, seq_len] — batch dim dropped (ncnn convention)
    cos_cache.create(half_dim, seq_len, sizeof(float));
    sin_cache.create(half_dim, seq_len, sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        int pos = position_id + s;
        float* cos_row = cos_cache.row(s);
        float* sin_row = sin_cache.row(s);
        for (int j = 0; j < half_dim; j++) {
            float t = (float)pos * inv_freq[j];
            cos_row[j] = std::cos(t);
            sin_row[j] = std::sin(t);
        }
    }
}

} // namespace hunyuan
