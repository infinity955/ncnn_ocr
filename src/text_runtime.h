#pragma once

#include <unordered_set>
#include <vector>

#include <mat.h>
#include <net.h>

#include "ocr_base.h"
#include "sampling.h"

// ============================================================
// Shared ncnn inference runtime helpers
// Used by all OCR models for the 4-module pipeline
// ============================================================

/// Batch text embedding lookup
ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, const std::vector<int>& input_ids);

/// Single-token text embedding lookup
ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, int token_id);

/// Run decoder with KV-cache support.
/// - is_prefill=true:  extracts new KV cache from decoder, appends to kv_cache
/// - is_prefill=false: feeds existing kv_cache as input, updates in-place
ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& decoder_net,
                                  const ncnn::Mat& embeds,
                                  const ncnn::Mat& mask,
                                  const ncnn::Mat& cos_cache,
                                  const ncnn::Mat& sin_cache,
                                  KVCache& kv_cache,
                                  int attn_cnt,
                                  bool is_prefill);

/// Run LM head projection
ncnn::Mat llm_run_lm_head(ncnn::Net& lm_head_net, const ncnn::Mat& hidden_states);