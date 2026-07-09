#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <cstdint>

#include <mat.h>
#include <net.h>

// ============================================================
// Common types shared by all OCR models
// ============================================================

using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;

struct GenerateConfig {
    int max_new_tokens = 256;
    float temperature = 0.0f;     // 0 = greedy
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int do_sample = 0;            // 0 = greedy, 1 = sample
};

// ============================================================
// OCRContext — base class for inference state
// ============================================================

class OCRContext {
public:
    virtual ~OCRContext() = default;
    virtual std::shared_ptr<OCRContext> clone() const = 0;

    KVCache kv_cache;
    int cur_token = 0;
    int position_id = 0;
};

class OCRBaseContext : public OCRContext {
public:
    std::shared_ptr<OCRContext> clone() const override {
        auto dst = std::make_shared<OCRBaseContext>();
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
// OCRBase — abstract interface for all OCR models
// ============================================================

class OCRBase {
public:
    virtual ~OCRBase() = default;

    virtual bool ok() const = 0;

    /// Full prefill: process image + prompt → first token + KV cache
    virtual std::shared_ptr<OCRContext> prefill(
        const std::string& prompt_text,
        const std::string& image_path) = 0;

    /// Autoregressive generation loop with KV-cache incremental decoding
    virtual void generate(
        std::shared_ptr<OCRContext> ctx,
        const GenerateConfig& cfg,
        std::function<void(const std::string&)> callback) = 0;
};

// ============================================================
// Factory function — auto-detect model type from model.json
// ============================================================

std::unique_ptr<OCRBase> create_ocr(const std::string& model_dir, int num_threads = 4);