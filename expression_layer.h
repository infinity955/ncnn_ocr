#pragma once

#include <net.h>
#include <string>
#include <vector>

namespace hunyuan {

/**
 * Custom ncnn layer to handle pnnx.Expression nodes.
 *
 * pnnx generates Expression nodes for operations that can't be
 * mapped to native ncnn layers. In HunYuanOCR, these are:
 *   1. Constants (0 inputs): expr=False, expr=6 → scalar constant
 *   2. Identity (1 input): expr=[@0] → pass-through
 *
 * The output size for constants is determined by the Split layer
 * that follows (stored in expr_output_size_).
 *
 * Usage:
 *   ncnn::Net net;
 *   net.register_custom_layer("pnnx.Expression", create_expression_layer);
 *   net.load_param(...);
 *   net.load_model(...);
 */
class ExpressionLayer : public ncnn::Layer
{
public:
    ExpressionLayer();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;

public:
    std::string expr;       // Expression string (e.g., "False", "6", "[@0]")
    int output_size;        // Output element count (for constants)

    // Pre-computed constant data (populated in load_param if expr is a constant)
    float constant_value;
};

/**
 * Pre-scan ncnn param file to determine Expression output sizes.
 * Finds Expression → Split chains and records the Split output count.
 *
 * Returns a map: Expression_layer_name → required_output_size
 */
std::vector<std::pair<std::string, int>> scan_expression_sizes(const char* param_path);

/**
 * Factory function for Expression layer registration.
 */
ncnn::Layer* create_expression_layer(void* userdata);

/**
 * Register Expression layer and pre-configure sizes for a net.
 * Call this BEFORE net.load_param().
 */
void register_expression_layer(ncnn::Net& net, const char* param_path);

/**
 * Custom ncnn layer to handle aten::to nodes.
 *
 * aten::to is a dtype/device conversion that is a no-op in ncnn
 * (everything is float32 on CPU). Simply passes input through.
 */
class AtenToLayer : public ncnn::Layer
{
public:
    AtenToLayer();

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;
};

/**
 * Factory function for aten::to layer registration.
 */
ncnn::Layer* create_atento_layer(void* userdata);

/**
 * Register aten::to pass-through layer for a net.
 * Call this BEFORE net.load_param().
 */
void register_atento_layer(ncnn::Net& net);

/**
 * Custom ncnn layer to handle Tensor.index nodes.
 *
 * In HunYuanOCR, Tensor.index is used for RoPE position indexing:
 * selects a row from a cos/sin table based on a position ID.
 *
 * Input 0: lookup table [dim, seq_len] (e.g., cos table 128x287)
 * Input 1: position index (scalar int)
 * Output: selected row [dim]
 */
class TensorIndexLayer : public ncnn::Layer
{
public:
    TensorIndexLayer();

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;
};

ncnn::Layer* create_tensorindex_layer(void* userdata);

void register_tensorindex_layer(ncnn::Net& net);

/**
 * Custom ncnn layer for torch.repeat_interleave.
 * Used for GQA KV-head expansion (8 KV heads → 16 Q heads).
 * Repeats each channel 'repeats' times along the channel (c) dimension.
 */
class RepeatInterleaveLayer : public ncnn::Layer
{
public:
    RepeatInterleaveLayer();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;

public:
    int repeats;
    int dim;  // PyTorch dim (-3 for heads in 4D tensor)
};

ncnn::Layer* create_repeat_interleave_layer(void* userdata);

void register_repeat_interleave_layer(ncnn::Net& net);

} // namespace hunyuan
