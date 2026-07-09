#pragma once

#include <array>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>

#include <mat.h>
#include <net.h>

#include "ocr_base.h"

// Forward declarations
class BpeTokenizer;

// ============================================================
// GLMOCR — GLM-OCR model implementation
// ============================================================

class GLMOCR : public OCRBase {
public:
    GLMOCR(const std::string& model_path, int num_threads = 4);
    ~GLMOCR();

    bool ok() const override { return ok_; }

    std::shared_ptr<OCRContext> prefill(const std::string& prompt_text,
                                        const std::string& image_path) override;

    void generate(std::shared_ptr<OCRContext> ctx,
                  const GenerateConfig& cfg,
                  std::function<void(const std::string&)> callback) override;

private:
    bool ok_ = false;
    int num_threads_ = 4;

    // ncnn networks
    std::shared_ptr<ncnn::Net> vision_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;

    // Tokenizer
    std::shared_ptr<BpeTokenizer> bpe_;
    std::unordered_set<int> additional_special_id_set_;

    // Model parameters
    int attn_cnt_ = 16;
    int hidden_size_ = 1536;
    int head_dim_ = 128;
    float rope_theta_ = 10000.0f;
    std::vector<int> mrope_section_;       // 3 entries: [t, h, w]
    int image_token_id_ = 59280;
    int patch_size_ = 14;
    int spatial_merge_size_ = 2;
    int vision_hidden_size_ = 1024;
    int vision_head_dim_ = 64;
    int vision_num_heads_ = 16;
    float vision_rope_theta_ = 0.0f;
    int vision_rope_dim_ = 0;
    std::vector<int> vision_mrope_section_; // 2 entries: [h_dim, w_dim]
    int max_num_patches_ = 3432;
    long long min_pixels_ = 12544;
    long long max_pixels_ = 9633792;
    float image_mean_[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std_[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    int eos_ = -1;
    int eop_ = -1;
    std::unordered_set<int> eos_ids_;   // full stop-token set (from generation_config eos_token_id)
    int vocab_size_ = 59392;

    // Image preprocessing helpers
    void get_image_size_for_patches(int img_h, int img_w, int& target_h, int& target_w) const;
    ncnn::Mat bgr_to_image_strip(const ncnn::Mat& bgr, int& num_patches_h, int& num_patches_w) const;

    // Vision encoder
    ncnn::Mat run_vision(const ncnn::Mat& image_strip, const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache) const;

    // Text RoPE generation
    void generate_text_rope_cache(int seq_len, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const;
};