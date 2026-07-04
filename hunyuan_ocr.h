#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <cstdint>

#include <net.h>

// Forward declare tokenizer (full definition in bpe_tokenizer.h)
class BpeTokenizer;

namespace hunyuan {

// ============================================================
// Configuration
// ============================================================

struct HunYuanConfig {
    // Model paths (relative to model_dir)
    std::string model_dir;

    // Model parameters
    int hidden_size = 1024;
    int num_layers = 24;
    int num_heads = 16;
    int num_kv_heads = 8;
    int head_dim = 128;
    int vocab_size = 120818;
    float rope_theta = 10000.0f;

    // Vision parameters
    int patch_size = 16;
    int spatial_merge_size = 2;
    int vision_hidden_size = 1152;

    // Special tokens
    int image_token_id = 120120;
    int image_start_token_id = 120118;
    int image_end_token_id = 120119;
    int image_newline_token_id = 120121;
    int bos_token_id = 120000;
    int eos_token_id = 120020;

    // Image preprocessing
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct GenerateConfig {
    int max_new_tokens = 256;
    float temperature = 0.0f;     // 0 = greedy
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    bool do_sample = false;
};

// ============================================================
// Context (state for generation loop)
// ============================================================

using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;

struct HunYuanContext {
    std::vector<int> token_ids;     // full sequence so far
    int cur_token = 0;              // last generated token
    int position_id = 0;            // next position
    ncnn::Mat vision_features;      // [1024, num_vision_tokens] — saved for re-injection
    std::vector<int> image_positions; // positions of image tokens in the prompt
    int num_vision_tokens = 0;
    KVCache kv_cache;               // KV cache for incremental decoding
};

// ============================================================
// Main OCR class
// ============================================================

class HunYuanOCR {
public:
    /**
     * Load models and tokenizer from model_dir.
     * model_dir should contain:
     *   model.json, vocab.json, merges.txt
     *   models/hunyuanocr_*.ncnn.{param,bin}
     */
    explicit HunYuanOCR(const std::string& model_dir, int num_threads = 4);
    ~HunYuanOCR();

    bool ok() const { return ok_; }

    // ---- Image preprocessing ----

    /**
     * Load image from file, resize to match ViT's traced grid dimensions,
     * normalize, extract patches, reorder for spatial merge.
     *
     * @param image_path  Path to image file (JPEG, PNG, etc.)
     * @param pixel_values Output: [num_patches, 3*patch_size*patch_size] = [1056, 768]
     * @return true on success
     */
    bool preprocess_image(const std::string& image_path, ncnn::Mat& pixel_values);

    // ---- Inference ----

    /**
     * Run vision encoder.
     * @param pixel_values  [num_patches, patch_dim] = [1056, 768]
     * @return image_embeds [num_vision_tokens, hidden_size]
     */
    ncnn::Mat run_vision_encoder(const ncnn::Mat& pixel_values,
                                  const ncnn::Mat& pos_embed);

    /**
     * Run text embedding.
     * @param token_ids  List of token IDs
     * @return text_embeddings [token_ids.size(), hidden_size]
     */
    ncnn::Mat run_text_embed(const std::vector<int>& token_ids);
    ncnn::Mat run_text_embed(int token_id);

    /**
     * Run text decoder (full prefill, no KV cache).
     * @param inputs_embeds  [seq_len, hidden_size]
     * @param causal_mask    [seq_len, seq_len] (upper triangle = -inf)
     * @param position_ids   [seq_len] int
     * @return hidden_states [seq_len, hidden_size]
     */
    ncnn::Mat run_text_decoder(const ncnn::Mat& inputs_embeds,
                                const ncnn::Mat& causal_mask,
                                const std::vector<int>& position_ids);

    /**
     * Run LM head.
     * @param hidden_states  [1, hidden_size] or [seq_len, hidden_size]
     * @return logits [1, vocab_size]
     */
    ncnn::Mat run_lm_head(const ncnn::Mat& hidden_states);

    /**
     * Full prefill: image + prompt → first token.
     * @param prompt  Text prompt (e.g., "检测并识别图片中的文字。")
     * @param image_path  Path to image
     * @return Context for generation loop
     */
    std::shared_ptr<HunYuanContext> prefill(const std::string& prompt,
                                             const std::string& image_path);

    /**
     * Generate loop (full prefill each step, no KV cache).
     * @param ctx  Context from prefill
     * @param cfg  Generation config
     * @param callback  Called with each decoded token string
     */
    void generate(std::shared_ptr<HunYuanContext> ctx,
                  const GenerateConfig& cfg,
                  std::function<void(const std::string&)> callback);

    // ---- Tokenizer ----
    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& ids) const;

    // ---- Helpers ----
    void build_prompt_and_tokenize(const std::string& user_prompt,
                                    int num_vision_tokens,
                                    std::vector<int>& token_ids,
                                    std::vector<int>& image_positions);

    int sample_token(const ncnn::Mat& logits,
                     const std::unordered_set<int>& history,
                     const GenerateConfig& cfg);

private:
    bool ok_ = false;
    int num_threads_ = 4;

    // ncnn networks
    std::shared_ptr<ncnn::Net> vision_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;

    // Tokenizer (defined in bpe_tokenizer.h)
    std::unique_ptr<BpeTokenizer> tokenizer_;

    // Special token IDs
    int bos_id_ = 120000;
    int eos_id_ = 120020;
    int image_token_id_ = 120120;
    int image_start_id_ = 120118;
    int image_end_id_ = 120119;

    // Model parameters
    int hidden_size_ = 1024;
    int num_layers_ = 24;
    int vocab_size_ = 120818;

    // Model directory
    std::string model_dir_;

    // Vision parameters
    int patch_size_ = 16;
    int spatial_merge_size_ = 2;

    // Current image dimensions (set by preprocess_image)
    int target_w_ = 0;
    int target_h_ = 0;
};

// ============================================================
// RoPE cos/sin generation (XD-RoPE for HunYuanOCR)
// ============================================================

/**
 * Generate cos/sin cache for XD-RoPE in interleaved format.
 * Matches HunYuanVLRotaryEmbedding output: [head_dim, seq_len]
 * where cos[i] = cos(position * inv_freq[i % half_dim]) for i < half_dim,
 * and cos[i] = cos[i - half_dim] (same value repeated for interleaved format).
 *
 * @param head_dim    Head dimension (128 for HunYuanOCR)
 * @param seq_len     Sequence length
 * @param position_id Starting position offset
 * @param rope_theta  RoPE theta (10000.0 for HunYuanOCR)
 */
void generate_xdrope_cache(int head_dim, int seq_len, int position_id,
                           ncnn::Mat& cos_cache, ncnn::Mat& sin_cache,
                           float rope_theta = 10000.0f);

} // namespace hunyuan
