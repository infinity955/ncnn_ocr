#include "paddleocr_vl.h"
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

struct PDVLContext : public OCRContext {
    std::shared_ptr<OCRContext> clone() const override {
        auto dst = std::make_shared<PDVLContext>();
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

// ============================================================ ctor
PaddleOCRVL::PaddleOCRVL(const std::string& model_path, int num_threads)
    : num_threads_(num_threads > 0 ? num_threads : 4) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs.is_open()) throw std::runtime_error("Cannot open model.json in " + model_path);
            ifs >> config;
        }
        vision_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();
        for (auto* net : {vision_net_.get(), text_embed_net_.get(), text_decoder_net_.get(), lm_head_net_.get()}) {
            net->opt.num_threads = num_threads_;
            net->opt.use_fp16_packed = false; net->opt.use_fp16_storage = false;
            net->opt.use_fp16_arithmetic = false; net->opt.use_bf16_storage = false;
        }
        auto P = [&](const char* k) { return model_path + "/" + config["params"][k].get<std::string>(); };
        printf("[PaddleOCRVL] Loading model from %s\n", model_path.c_str());
        vision_net_->load_param(P("vision_param").c_str());   vision_net_->load_model(P("vision_bin").c_str());
        text_embed_net_->load_param(P("text_embed_param").c_str()); text_embed_net_->load_model(P("text_embed_bin").c_str());
        text_decoder_net_->load_param(P("text_decoder_param").c_str()); text_decoder_net_->load_model(P("text_decoder_bin").c_str());
        lm_head_net_->load_param(P("lm_head_param").c_str());  lm_head_net_->load_model(P("lm_head_bin").c_str());

        // tokenizer (SentencePiece-style BPE: type != bbpe -> no byte_encoder; SP decode + byte_fallback)
        std::string type = config["tokenizer"].value("type", "bpe");
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();
        bpe_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"));
        if (config["tokenizer"].contains("additional_special_tokens")) {
            for (const auto& t : config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>())
                bpe_->AddAdditionalSpecialToken(t);
        }
        for (int id : bpe_->additional_special_token_ids()) additional_special_id_set_.insert(id);

        // stop tokens
        if (config["tokenizer"].contains("eos")) {
            auto e = config["tokenizer"]["eos"].get<std::string>();
            auto it = bpe_->token_to_id().find(e);
            if (it != bpe_->token_to_id().end()) eos_ids_.insert(it->second);
        }
        if (config["tokenizer"].contains("eos_ids"))
            for (int id : config["tokenizer"]["eos_ids"].get<std::vector<int>>()) eos_ids_.insert(id);

        auto& s = config["setting"];
        attn_cnt_ = s.value("attn_cnt", attn_cnt_);
        hidden_size_ = s.value("hidden_size", hidden_size_);
        head_dim_ = s.value("head_dim", head_dim_);
        image_token_id_ = s.value("image_token_id", image_token_id_);
        {
            auto& r = s["rope"];
            rope_theta_ = r.value("rope_theta", rope_theta_);
            head_dim_ = r.value("rope_head_dim", head_dim_);
            for (auto& v : r["mrope_section"]) mrope_section_.push_back(v.get<int>());
        }
        auto& v = s["vision"];
        patch_size_ = v.value("patch_size", patch_size_);
        spatial_merge_size_ = v.value("spatial_merge_size", spatial_merge_size_);
        vision_hidden_size_ = v.value("vision_hidden_size", vision_hidden_size_);
        vision_head_dim_ = v.value("vision_head_dim", vision_head_dim_);
        vision_num_heads_ = v.value("vision_num_heads", vision_num_heads_);
        pos_embed_grid_ = v.value("pos_embed_base_grid", pos_embed_grid_);
        min_pixels_ = v.value("min_pixels", min_pixels_);
        max_pixels_ = v.value("max_pixels", max_pixels_);
        {
            auto& vr = v["rope"];
            vision_rope_theta_ = vr.value("rope_theta", vision_rope_theta_);
            vision_rope_dim_ = vr.value("rope_head_dim", vision_rope_dim_);
            if (vr.contains("mrope_section"))
                for (auto& x : vr["mrope_section"]) vision_mrope_section_.push_back(x.get<int>());
        }
        if (vision_mrope_section_.size() != 2)  // default [rdim/2, rdim/2]
            vision_mrope_section_ = {vision_rope_dim_ / 2, vision_rope_dim_ / 2};
        if (v.contains("image_mean")) { auto m = v["image_mean"].get<std::vector<float>>(); for (int i=0;i<3;i++) image_mean_[i]=m[i]; }
        if (v.contains("image_std")) { auto m = v["image_std"].get<std::vector<float>>(); for (int i=0;i<3;i++) image_std_[i]=m[i]; }

        // learned position-embedding base (27x27x1152) for bilinear interpolation
        std::string pe_path = model_path + "/" + config["params"].value("pos_embed", std::string("models/pdvl_pos_embed.bin"));
        {
            std::ifstream f(pe_path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open pos_embed: " + pe_path);
            size_t n = (size_t)pos_embed_grid_ * pos_embed_grid_ * vision_hidden_size_;
            pos_embed_base_.resize(n);
            f.read((char*)pos_embed_base_.data(), n * sizeof(float));
            if ((size_t)f.gcount() != n * sizeof(float)) throw std::runtime_error("pos_embed size mismatch");
        }

        vocab_size_ = (int)bpe_->vocab_size();
        printf("  attn_cnt=%d hidden=%d head_dim=%d vocab=%d image_token=%d\n",
               attn_cnt_, hidden_size_, head_dim_, vocab_size_, image_token_id_);
        printf("  text mrope=[%d %d %d] theta=%.0f; vision hidden=%d depth-rope theta=%.0f rope_dim=%d sec=[%d %d]\n",
               mrope_section_[0], mrope_section_[1], mrope_section_[2], rope_theta_,
               vision_hidden_size_, vision_rope_theta_, vision_rope_dim_,
               vision_mrope_section_[0], vision_mrope_section_[1]);
        ok_ = true;
    } catch (std::exception& e) {
        fprintf(stderr, "[PaddleOCRVL] Load failed: %s\n", e.what());
        ok_ = false;
    }
}

PaddleOCRVL::~PaddleOCRVL() = default;

// ============================================================ preprocessing
void PaddleOCRVL::get_image_size_for_patches(int img_h, int img_w, int& target_h, int& target_w) const {
    int eff = patch_size_ * spatial_merge_size_;   // 28
    auto round_by = [&](double sz) { return std::max(eff, (int)(std::round(sz / (double)eff) * eff)); };
    double h = img_h, w = img_w, area = h * w, scale = 1.0;
    if (area > (double)max_pixels_) scale = std::sqrt((double)max_pixels_ / area);
    else if (area < (double)min_pixels_) scale = std::sqrt((double)min_pixels_ / area);
    target_h = round_by(h * scale);
    target_w = round_by(w * scale);
}

// block-major single-frame strip (1,3,14,14*N), CLIP-normalized RGB — matches modules.py strip
ncnn::Mat PaddleOCRVL::bgr_to_image_strip(const ncnn::Mat& bgr, int& nph, int& npw) const {
    int target_h, target_w;
    get_image_size_for_patches(bgr.h, bgr.w, target_h, target_w);
    ncnn::Mat r = ncnn_mat_resize(bgr, target_w, target_h);
    nph = target_h / patch_size_; npw = target_w / patch_size_;
    int N = nph * npw;
    const unsigned char* data = (const unsigned char*)r.data;
    ncnn::Mat strip(patch_size_ * N, patch_size_, 3); strip.fill(0.0f);
    int gh = nph / spatial_merge_size_, gw = npw / spatial_merge_size_, idx = 0;
    for (int bh = 0; bh < gh; bh++) for (int bw = 0; bw < gw; bw++)
        for (int mh = 0; mh < spatial_merge_size_; mh++) for (int mw = 0; mw < spatial_merge_size_; mw++) {
            int ph = bh * spatial_merge_size_ + mh, pw = bw * spatial_merge_size_ + mw;
            int sy = ph * patch_size_, sx = pw * patch_size_, bx = idx * patch_size_;
            for (int y = 0; y < patch_size_; y++) {
                const unsigned char* ir = data + (sy + y) * target_w * 3;
                float* dr = strip.channel(0).row(y) + bx;
                float* dg = strip.channel(1).row(y) + bx;
                float* db = strip.channel(2).row(y) + bx;
                for (int x = 0; x < patch_size_; x++) {
                    const unsigned char* px = ir + (sx + x) * 3;
                    dr[x] = (px[2] / 255.0f - image_mean_[0]) / image_std_[0];
                    dg[x] = (px[1] / 255.0f - image_mean_[1]) / image_std_[1];
                    db[x] = (px[0] / 255.0f - image_mean_[2]) / image_std_[2];
                }
            }
            idx++;
        }
    return strip;
}

// bilinear-interpolate 27x27 pos-embed base -> (nph,npw), block-major order. Matches
// HF F.interpolate(bilinear, align_corners=False). Returns Mat(1152, N).
ncnn::Mat PaddleOCRVL::interpolate_pos_embed(int nph, int npw) const {
    int C = vision_hidden_size_, G = pos_embed_grid_, N = nph * npw;
    ncnn::Mat pe(C, N);
    double sh = (double)G / nph, sw = (double)G / npw;
    int gh = nph / spatial_merge_size_, gw = npw / spatial_merge_size_, idx = 0;
    auto clampi = [&](int a) { return a < 0 ? 0 : (a > G - 1 ? G - 1 : a); };
    for (int bh = 0; bh < gh; bh++) for (int bw = 0; bw < gw; bw++)
        for (int mh = 0; mh < spatial_merge_size_; mh++) for (int mw = 0; mw < spatial_merge_size_; mw++) {
            int h = bh * spatial_merge_size_ + mh, w = bw * spatial_merge_size_ + mw;
            double fy = std::max(0.0, (h + 0.5) * sh - 0.5);
            double fx = std::max(0.0, (w + 0.5) * sw - 0.5);
            int y0 = (int)std::floor(fy), x0 = (int)std::floor(fx);
            double wy = fy - y0, wx = fx - x0;
            int y1 = clampi(y0 + 1), x1 = clampi(x0 + 1); y0 = clampi(y0); x0 = clampi(x0);
            const float* p00 = &pos_embed_base_[(size_t)(y0 * G + x0) * C];
            const float* p01 = &pos_embed_base_[(size_t)(y0 * G + x1) * C];
            const float* p10 = &pos_embed_base_[(size_t)(y1 * G + x0) * C];
            const float* p11 = &pos_embed_base_[(size_t)(y1 * G + x1) * C];
            float* dst = pe.row(idx);
            for (int c = 0; c < C; c++)
                dst[c] = (float)((1 - wy) * ((1 - wx) * p00[c] + wx * p01[c]) +
                                 wy * ((1 - wx) * p10[c] + wx * p11[c]));
            idx++;
        }
    return pe;
}

ncnn::Mat PaddleOCRVL::run_vision(const ncnn::Mat& strip, const ncnn::Mat& pos_embed,
                                  const ncnn::Mat& cos, const ncnn::Mat& sin) const {
    ncnn::Mat out;
    ncnn::Extractor ex = vision_net_->create_extractor();
    ex.input("in0", strip); ex.input("in1", pos_embed); ex.input("in2", cos); ex.input("in3", sin);
    ex.extract("out0", out);
    return out;
}

// ============================================================ prefill
std::shared_ptr<OCRContext> PaddleOCRVL::prefill(const std::string& prompt_text, const std::string& image_path) {
    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) { fprintf(stderr, "[PaddleOCRVL] Failed to load image: %s\n", image_path.c_str()); return nullptr; }
    printf("[PaddleOCRVL] Image: %dx%d\n", bgr.w, bgr.h);

    int nph = 0, npw = 0;
    ncnn::Mat strip = bgr_to_image_strip(bgr, nph, npw);
    ncnn::Mat pos_embed = interpolate_pos_embed(nph, npw);
    ncnn::Mat vcos, vsin;
    generate_vision_rope_cache_2d(nph, npw, spatial_merge_size_, vision_rope_theta_,
                                  vision_mrope_section_, false, vcos, vsin);
    ncnn::Mat vision_features = run_vision(strip, pos_embed, vcos, vsin);
    int num_vision_tokens = vision_features.h;
    printf("[PaddleOCRVL] Vision tokens: %d (grid %dx%d)\n", num_vision_tokens, nph, npw);

    // chat template: <|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>xN<|IMAGE_END|>{prompt}\nAssistant:\n
    std::string fp = "<|begin_of_sentence|>User: <|IMAGE_START|>";
    for (int i = 0; i < num_vision_tokens; i++) fp += "<|IMAGE_PLACEHOLDER|>";
    fp += "<|IMAGE_END|>" + prompt_text + "\nAssistant:\n";

    std::vector<int> token_ids = bpe_->encode(fp, false, false);
    printf("[PaddleOCRVL] Prompt: %zu tokens\n", token_ids.size());

    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids).clone();
    std::vector<int> img_pos;
    for (int i = 0; i < (int)token_ids.size(); i++) if (token_ids[i] == image_token_id_) img_pos.push_back(i);
    if ((int)img_pos.size() == num_vision_tokens) {
        for (int i = 0; i < num_vision_tokens; i++)
            memcpy(token_embed.row(img_pos[i]), vision_features.row(i), hidden_size_ * sizeof(float));
    } else {
        printf("[PaddleOCRVL] WARNING: image token mismatch vision=%d prompt=%d\n", num_vision_tokens, (int)img_pos.size());
    }

    int seq_len = (int)token_ids.size();
    ncnn::Mat mask(seq_len, seq_len); mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) { float* row = mask.row(i); for (int j = i + 1; j < seq_len; j++) row[j] = -1e38f; }

    ncnn::Mat cos_cache, sin_cache;
    int next_position_id = seq_len;
    if (!img_pos.empty()) {
        generate_rope_embed_cache_vision_mrope(seq_len, head_dim_, 0, img_pos[0], num_vision_tokens,
                                               npw, spatial_merge_size_, mrope_section_,
                                               cos_cache, sin_cache, rope_theta_);
        next_position_id = seq_len - num_vision_tokens + (npw / spatial_merge_size_);
    } else {
        generate_rope_embed_cache(seq_len, head_dim_, 0, cos_cache, sin_cache, rope_theta_);
    }

    KVCache kv_cache;
    ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask,
                                                   cos_cache, sin_cache, kv_cache, attn_cnt_, true);
    ncnn::Mat last_hidden = decode_out.row_range(seq_len - 1, 1);
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);
    int next_token_id = argmax1d(logits);
    printf("[PaddleOCRVL] Prefill done, next_token=%d\n", next_token_id);

    auto ctx = std::make_shared<PDVLContext>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = next_position_id;
    return ctx;
}

// ============================================================ generate
void PaddleOCRVL::generate(std::shared_ptr<OCRContext> ctx_base, const GenerateConfig& cfg,
                           std::function<void(const std::string&)> callback) {
    auto ctx = std::static_pointer_cast<PDVLContext>(ctx_base);
    std::unordered_set<int> history; history.insert(ctx->cur_token);
    std::string emitted;
    LlmTokenSampleConfig sc;
    sc.vocab_size = vocab_size_; sc.temperature = cfg.temperature; sc.top_p = cfg.top_p;
    sc.top_k = cfg.top_k; sc.repetition_penalty = cfg.repetition_penalty; sc.do_sample = cfg.do_sample;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        bool is_stop = eos_ids_.count(ctx->cur_token);
        bool is_special = is_stop || additional_special_id_set_.count(ctx->cur_token);
        // PaddleOCR-VL outputs <|LOC_d1|><|LOC_d2|>... coordinate tokens mixed with text.
        // Filter them so the user sees clean text. Also skip byte-level newline tokens.
        std::string t = bpe_->decode({ctx->cur_token}, false);
        bool is_loc = (t.size() >= 5 && t.substr(0, 5) == "<|LOC") ||
                      (t == "\\n" || t == "\n");
        // SentencePiece metaspace prefix -> space for readability
        if (!t.empty() && (unsigned char)t[0] == 0xe2 && t.size() >= 3 &&
            (unsigned char)t[1] == 0x96 && (unsigned char)t[2] == 0x81) {
            t = " " + t.substr(3);  // "▁" -> " "
        }
        if (!is_special && !is_loc) {
            emitted += t;
            callback(t);
        }
        if (is_stop) break;
        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, ctx->cur_token);
        ncnn::Mat cos_cache, sin_cache;
        generate_rope_embed_cache(1, head_dim_, ctx->position_id, cos_cache, sin_cache, rope_theta_);
        ctx->position_id++;
        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1); mask.fill(0.0f);
        ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, cur_embed, mask,
                                                       cos_cache, sin_cache, ctx->kv_cache, attn_cnt_, false);
        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);
        int next_id = llm_select_next_token(logits, history, sc);
        ctx->cur_token = next_id;
        history.insert(next_id);
    }
}
