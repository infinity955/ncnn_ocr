#pragma once

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

class BpeTokenizer;

// ============================================================
// PaddleOCRVL — PaddleOCR-VL-1.6 (SigLIP vision + ERNIE-4.5 decoder)
// ============================================================
class PaddleOCRVL : public OCRBase {
public:
    PaddleOCRVL(const std::string& model_path, int num_threads = 4);
    ~PaddleOCRVL();

    bool ok() const override { return ok_; }

    std::shared_ptr<OCRContext> prefill(const std::string& prompt_text,
                                        const std::string& image_path) override;
    void generate(std::shared_ptr<OCRContext> ctx, const GenerateConfig& cfg,
                  std::function<void(const std::string&)> callback) override;

private:
    bool ok_ = false;
    int num_threads_ = 4;

    std::shared_ptr<ncnn::Net> vision_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;
    std::shared_ptr<BpeTokenizer> bpe_;
    std::unordered_set<int> additional_special_id_set_;

    // text decoder
    int attn_cnt_ = 18;
    int hidden_size_ = 1024;
    int head_dim_ = 128;
    float rope_theta_ = 500000.0f;
    std::vector<int> mrope_section_;              // [16,24,24]
    int image_token_id_ = 100295;
    int vocab_size_ = 103424;
    std::unordered_set<int> eos_ids_;

    // vision
    int patch_size_ = 14;
    int spatial_merge_size_ = 2;
    int vision_hidden_size_ = 1152;
    int vision_head_dim_ = 72;
    int vision_num_heads_ = 16;
    float vision_rope_theta_ = 10000.0f;
    int vision_rope_dim_ = 36;
    std::vector<int> vision_mrope_section_;        // [18,18]
    int pos_embed_grid_ = 27;
    std::vector<float> pos_embed_base_;            // (27*27, 1152) raster

    long long min_pixels_ = 101920;
    long long max_pixels_ = 1003520;
    float image_mean_[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std_[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    void get_image_size_for_patches(int img_h, int img_w, int& target_h, int& target_w) const;
    ncnn::Mat bgr_to_image_strip(const ncnn::Mat& bgr, int& num_patches_h, int& num_patches_w) const;
    ncnn::Mat interpolate_pos_embed(int num_patches_h, int num_patches_w) const;  // block-major (1152, N)
    ncnn::Mat run_vision(const ncnn::Mat& strip, const ncnn::Mat& pos_embed,
                         const ncnn::Mat& cos, const ncnn::Mat& sin) const;
};
