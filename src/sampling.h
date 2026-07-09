#pragma once

#include <unordered_set>
#include <vector>

#include <mat.h>

// ============================================================
// Shared token sampling utilities
// ============================================================

struct LlmTokenSampleConfig {
    int vocab_size = 0;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    int do_sample = 0;
};

/// In-place softmax with temperature scaling
void softmax_vec(std::vector<float>& logits, float temperature);

/// Zero out probabilities below the top-k threshold
void apply_top_k(std::vector<float>& probs, int k);

/// Nucleus sampling: keep smallest set with cumulative probability >= p
void apply_top_p(std::vector<float>& probs, float p);

/// Sample from a probability distribution
int sample_from_probs(const std::vector<float>& probs);

/// Greedy argmax over a 1D ncnn::Mat
inline int argmax1d(const ncnn::Mat& m) {
    const float* p = m;
    int max_idx = 0;
    float max_val = p[0];
    for (int i = 1; i < m.w; ++i) {
        if (p[i] > max_val) {
            max_val = p[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/// Full sampling pipeline: repetition penalty → greedy or (softmax → top-k → top-p → sample)
int llm_select_next_token(const ncnn::Mat& logits,
                          const std::unordered_set<int>& history,
                          const LlmTokenSampleConfig& cfg);