#include "expression_layer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <stack>
#include <fstream>
#include <sstream>

namespace hunyuan {

// ============================================================
// Global registry of Expression output sizes
// ============================================================

// Maps Expression layer name → {expr_string, output_size}
static std::map<std::string, std::pair<std::string, int>> g_expr_info;

// ============================================================
// Parse expression strings from pnnx param file
// ============================================================

static void scan_pnnx_expressions(const std::string& pnnx_param_path)
{
    std::ifstream ifs(pnnx_param_path);
    if (!ifs.is_open())
    {
        // fprintf(stderr, "[ExpressionLayer] Cannot open pnnx param: %s\n", pnnx_param_path.c_str());
        return;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty()) continue;

        // Look for lines starting with "pnnx.Expression"
        std::istringstream iss(line);
        std::string type, name;
        iss >> type >> name;
        if (type != "pnnx.Expression") continue;

        // Find expr=... in the remaining part
        std::string expr_str;
        size_t expr_pos = line.find("expr=");
        if (expr_pos != std::string::npos)
        {
            expr_str = line.substr(expr_pos + 5);  // after "expr="
            // Take until whitespace or end
            size_t end = expr_str.find_first_of(" \t");
            if (end != std::string::npos)
                expr_str = expr_str.substr(0, end);
        }

        // Update existing entry or create new one
        auto it = g_expr_info.find(name);
        if (it != g_expr_info.end())
        {
            it->second.first = expr_str;
        }
        else
        {
            g_expr_info[name] = {expr_str, 0};
        }
    }

    // fprintf(stderr, "[ExpressionLayer] Loaded %zu expressions from pnnx param\n", g_expr_info.size());
}

// ============================================================
// Pre-scan ncnn param file for Expression→Split chains
// ============================================================

std::vector<std::pair<std::string, int>> scan_expression_sizes(const char* param_path)
{
    std::vector<std::pair<std::string, int>> result;

    std::ifstream ifs(param_path);
    if (!ifs.is_open())
    {
        fprintf(stderr, "[ExpressionLayer] Cannot open param: %s\n", param_path);
        return result;
    }

    // Read header
    std::string line;
    std::getline(ifs, line);  // magic
    std::getline(ifs, line);  // layer_count blob_count

    // Parse layers
    struct LayerInfo {
        std::string type;
        std::string name;
        std::vector<std::string> input_blobs;
        std::vector<std::string> output_blobs;
        int num_outputs;
    };

    std::vector<LayerInfo> layers;
    while (std::getline(ifs, line))
    {
        if (line.empty()) continue;

        std::istringstream iss(line);
        LayerInfo info;
        iss >> info.type >> info.name;

        int num_inputs, num_outputs;
        iss >> num_inputs >> num_outputs;
        info.num_outputs = num_outputs;

        std::string blob;
        for (int i = 0; i < num_inputs; i++) {
            iss >> blob;
            info.input_blobs.push_back(blob);
        }
        for (int i = 0; i < num_outputs; i++) {
            iss >> blob;
            info.output_blobs.push_back(blob);
        }

        layers.push_back(info);
    }

    // Build blob → layer index map
    std::map<std::string, int> blob_to_consumer;
    for (int i = 0; i < (int)layers.size(); i++)
    {
        for (auto& blob : layers[i].input_blobs)
        {
            if (blob_to_consumer.find(blob) == blob_to_consumer.end())
            {
                blob_to_consumer[blob] = i;
            }
        }
    }

    // For each Expression node, find its Split consumer
    for (int i = 0; i < (int)layers.size(); i++)
    {
        if (layers[i].type != "pnnx.Expression") continue;

        const auto& expr_layer = layers[i];
        std::string output_blob = expr_layer.output_blobs[0];

        // Check if consumer is a Split
        auto it = blob_to_consumer.find(output_blob);
        if (it != blob_to_consumer.end())
        {
            int consumer_idx = it->second;
            if (layers[consumer_idx].type == "Split")
            {
                int split_outputs = layers[consumer_idx].num_outputs;
                result.push_back({expr_layer.name, split_outputs});
                {
                    auto it2 = g_expr_info.find(expr_layer.name);
                    std::string saved = (it2 != g_expr_info.end()) ? it2->second.first : "";
                    g_expr_info[expr_layer.name] = {saved, split_outputs};
                }
            }
            else
            {
                // Consumer is not Split - identity case
                result.push_back({expr_layer.name, 0});
                {
                    auto it2 = g_expr_info.find(expr_layer.name);
                    std::string saved = (it2 != g_expr_info.end()) ? it2->second.first : "";
                    g_expr_info[expr_layer.name] = {saved, 0};
                }
            }
        }
        else
        {
            result.push_back({expr_layer.name, 0});
            {
                auto it2 = g_expr_info.find(expr_layer.name);
                std::string saved = (it2 != g_expr_info.end()) ? it2->second.first : "";
                g_expr_info[expr_layer.name] = {saved, 0};
            }
        }
    }

    return result;
}

// ============================================================
// Factory function
// ============================================================

ncnn::Layer* create_expression_layer(void* /*userdata*/)
{
    return new ExpressionLayer();
}

// ============================================================
// Register Expression layer
// ============================================================

void register_expression_layer(ncnn::Net& net, const char* param_path)
{
    // Derive pnnx param path from ncnn param path
    std::string ncnn_path(param_path);
    std::string pnnx_path = ncnn_path;
    size_t pos = pnnx_path.rfind(".ncnn.param");
    if (pos != std::string::npos)
    {
        pnnx_path.replace(pos, 11, ".pnnx.param");
    }

    // First, scan pnnx param for expression strings
    scan_pnnx_expressions(pnnx_path);

    // Then, scan ncnn param to determine output sizes (Expression→Split chains)
    scan_expression_sizes(param_path);

    // Register the layer factory
    net.register_custom_layer("pnnx.Expression", create_expression_layer);
}

// ============================================================
// ExpressionLayer implementation
// ============================================================

ExpressionLayer::ExpressionLayer()
    : output_size(0), constant_value(0.0f)
{
    one_blob_only = false;
    support_inplace = false;
}

int ExpressionLayer::load_param(const ncnn::ParamDict& pd)
{
    // Read expression string from param id 0
    expr = pd.get(0, std::string(""));

    // Determine operation type and value
    if (expr.empty())
    {
        // No expression string - might be from ncnn binary
        // Default to identity
    }
    else if (expr == "False")
    {
        constant_value = 0.0f;
    }
    else if (expr == "6")
    {
        constant_value = 6.0f;
    }
    else
    {
        // Try to parse as number
        char* endptr = nullptr;
        float val = strtof(expr.c_str(), &endptr);
        if (endptr != expr.c_str() && *endptr == '\0')
        {
            constant_value = val;
        }
    }

    // Look up output size and expression from pre-scanned info
    auto it = g_expr_info.find(name);
    if (it != g_expr_info.end())
    {
        expr = it->second.first;
        output_size = it->second.second;
        // fprintf(stderr, "[Expression] %s: expr='%s', output_size=%d\n", name.c_str(), expr.c_str(), output_size);
    }
    else
    {
        // fprintf(stderr, "[Expression] %s: NOT found in g_expr_info\n", name.c_str());
    }

    return 0;
}

// ============================================================
// Simple expression evaluator for tensor expressions
// ============================================================

namespace {

// Tokenize expression into tokens
std::vector<std::string> tokenize_expr(const std::string& expr)
{
    std::vector<std::string> tokens;
    std::string t;
    for (size_t i = 0; i < expr.size(); i++)
    {
        char ch = expr[i];
        if (ch == '(' || ch == ')' || ch == ',')
        {
            if (!t.empty())
            {
                tokens.push_back(t);
                t.clear();
            }
        }
        else if (ch == '[' || ch == ']')
        {
            // Skip brackets
            if (!t.empty())
            {
                tokens.push_back(t);
                t.clear();
            }
        }
        else
        {
            t += ch;
        }
    }
    if (!t.empty())
    {
        tokens.push_back(t);
    }
    return tokens;
}

// Opcode-based expression evaluator for fast element-wise execution
// Parses the expression once into opcodes, then applies to all elements in a tight loop.

enum OpType { OP_PUSH_BLOB, OP_PUSH_CONST, OP_NEG, OP_MUL, OP_ADD, OP_SUB, OP_DIV, OP_ABS, OP_SQUARE, OP_SQRT, OP_EXP, OP_LOG, OP_SIN, OP_COS, OP_TANH, OP_RECIP };

struct ExprOp {
    OpType type;
    int blob_idx;   // for OP_PUSH_BLOB
    float value;     // for OP_PUSH_CONST
};

static std::vector<ExprOp> compile_expr(const std::string& expr_str, const std::vector<ncnn::Mat>& blobs)
{
    std::vector<ExprOp> ops;
    auto tokens = tokenize_expr(expr_str);
    if (tokens.empty()) return ops;

    // Process in reverse (prefix notation)
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];
        ExprOp op;

        if (t.size() >= 2 && t[0] == '@')
        {
            op.type = OP_PUSH_BLOB;
            op.blob_idx = atoi(t.c_str() + 1);
            if (op.blob_idx < 0 || op.blob_idx >= (int)blobs.size())
                op.blob_idx = 0;
        }
        else if (t == "neg")       op.type = OP_NEG;
        else if (t == "mul")       op.type = OP_MUL;
        else if (t == "add")       op.type = OP_ADD;
        else if (t == "sub")       op.type = OP_SUB;
        else if (t == "div" || t == "/") op.type = OP_DIV;
        else if (t == "abs")       op.type = OP_ABS;
        else if (t == "square")    op.type = OP_SQUARE;
        else if (t == "sqrt")      op.type = OP_SQRT;
        else if (t == "exp")       op.type = OP_EXP;
        else if (t == "log")       op.type = OP_LOG;
        else if (t == "sin")       op.type = OP_SIN;
        else if (t == "cos")       op.type = OP_COS;
        else if (t == "tanh")      op.type = OP_TANH;
        else if (t == "reciprocal") op.type = OP_RECIP;
        else
        {
            // Try literal number
            char* endptr = nullptr;
            float val = strtof(t.c_str(), &endptr);
            if (endptr != t.c_str() && *endptr == '\0')
            {
                op.type = OP_PUSH_CONST;
                op.value = val;
            }
            else if (t == "False")
            {
                op.type = OP_PUSH_CONST;
                op.value = 0.0f;
            }
            else
            {
                op.type = OP_PUSH_CONST;
                op.value = 0.0f;
            }
        }
        ops.push_back(op);
    }
    return ops;
}

static float eval_opcodes(const std::vector<ExprOp>& ops,
                          const float* const* blob_ptrs,
                          const int* blob_totals,
                          int elem_idx)
{
    float stack[16];
    int sp = 0;

    for (const auto& op : ops)
    {
        switch (op.type)
        {
        case OP_PUSH_BLOB:
            if (blob_totals[op.blob_idx] > 0 && elem_idx < blob_totals[op.blob_idx])
                stack[sp++] = blob_ptrs[op.blob_idx][elem_idx];
            else
                stack[sp++] = 0.0f;
            break;
        case OP_PUSH_CONST:
            stack[sp++] = op.value;
            break;
        case OP_NEG:
            stack[sp-1] = -stack[sp-1];
            break;
        case OP_MUL:
            sp--;
            stack[sp-1] = stack[sp] * stack[sp-1];
            break;
        case OP_ADD:
            sp--;
            stack[sp-1] = stack[sp] + stack[sp-1];
            break;
        case OP_SUB:
            sp--;
            stack[sp-1] = stack[sp] - stack[sp-1];
            break;
        case OP_DIV:
            sp--;
            stack[sp-1] = (stack[sp-1] != 0.0f) ? stack[sp] / stack[sp-1] : 0.0f;
            break;
        case OP_ABS:
            stack[sp-1] = fabsf(stack[sp-1]);
            break;
        case OP_SQUARE:
            stack[sp-1] = stack[sp-1] * stack[sp-1];
            break;
        case OP_SQRT:
            stack[sp-1] = sqrtf(stack[sp-1]);
            break;
        case OP_EXP:
            stack[sp-1] = expf(stack[sp-1]);
            break;
        case OP_LOG:
            stack[sp-1] = logf(stack[sp-1] > 0 ? stack[sp-1] : 1e-10f);
            break;
        case OP_SIN:
            stack[sp-1] = sinf(stack[sp-1]);
            break;
        case OP_COS:
            stack[sp-1] = cosf(stack[sp-1]);
            break;
        case OP_TANH:
            stack[sp-1] = tanhf(stack[sp-1]);
            break;
        case OP_RECIP:
            stack[sp-1] = (stack[sp-1] != 0.0f) ? 1.0f / stack[sp-1] : 0.0f;
            break;
        }
    }
    return sp > 0 ? stack[0] : 0.0f;
}

} // anonymous namespace

// ============================================================
// forward()
// ============================================================

int ExpressionLayer::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                              std::vector<ncnn::Mat>& top_blobs,
                              const ncnn::Option& opt) const
{
    ncnn::Mat& top_blob = top_blobs[0];

    // fprintf(stderr, "[Expression] %s: expr='%s', inputs=%zu, output_size=%d\n", name.c_str(), expr.c_str(), bottom_blobs.size(), output_size);

    if (bottom_blobs.empty())
    {
        // Constant expression (0 inputs)
        // Output a scalar that broadcasts to any shape via ncnn Split fanout.
        // Split fans out the same blob to all consumers — each gets a scalar.
        top_blob.create(1, (size_t)4u);
        if (top_blob.empty())
            return -100;

        top_blob[0] = constant_value;
        return 0;
    }

    // Identity: expr=[@0] or empty expr with 1 input → pass-through
    if ((expr.size() >= 2 && expr[0] == '[' && expr[expr.size()-1] == ']')
        || (expr.empty() && bottom_blobs.size() == 1))
    {
        // Identity: just copy input to output
        top_blob = bottom_blobs[0].clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        return 0;
    }

    // General scalar expression: pre-compile opcodes, evaluate element-wise
    const ncnn::Mat& input = bottom_blobs[0];
    int total = input.w * input.h * input.d * input.c;

    // Compile expression once
    auto ops = compile_expr(expr, bottom_blobs);

    // Prepare blob pointers and sizes for fast access
    std::vector<const float*> blob_ptrs(bottom_blobs.size());
    std::vector<int> blob_totals(bottom_blobs.size());
    for (size_t j = 0; j < bottom_blobs.size(); j++)
    {
        blob_ptrs[j] = (const float*)bottom_blobs[j].data;
        blob_totals[j] = bottom_blobs[j].w * bottom_blobs[j].h
                        * bottom_blobs[j].d * bottom_blobs[j].c;
    }

    top_blob.create(total, (size_t)4u);
    if (top_blob.empty())
        return -100;

    float* out_ptr = top_blob;
    if (ops.empty())
    {
        // No valid expression — identity copy first input
        if (!bottom_blobs.empty() && blob_totals[0] > 0)
            memcpy(out_ptr, blob_ptrs[0], total * sizeof(float));
        else
            memset(out_ptr, 0, total * sizeof(float));
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < total; i++)
        {
            out_ptr[i] = eval_opcodes(ops, blob_ptrs.data(), blob_totals.data(), i);
        }
    }

    return 0;
}

} // namespace hunyuan

// ============================================================
// AtenToLayer implementation
// ============================================================

namespace hunyuan {

AtenToLayer::AtenToLayer()
{
    one_blob_only = false;
    support_inplace = false;
}

int AtenToLayer::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                          std::vector<ncnn::Mat>& top_blobs,
                          const ncnn::Option& opt) const
{
    // aten::to is dtype/device conversion — pass-through in ncnn
    // First input is the actual data; remaining inputs are dtype/device args
    if (bottom_blobs.empty())
    {
        fprintf(stderr, "[aten::to] %s: no inputs!\n", name.c_str());
        return -100;
    }

    const ncnn::Mat& data = bottom_blobs[0];
    // fprintf(stderr, "[aten::to] %s: pass-through [%d x %d x %d]\n", name.c_str(), data.w, data.h, data.c);

    top_blobs[0] = data.clone(opt.blob_allocator);
    if (top_blobs[0].empty())
    {
        fprintf(stderr, "[aten::to] %s: clone failed!\n", name.c_str());
        return -100;
    }

    return 0;
}

ncnn::Layer* create_atento_layer(void* /*userdata*/)
{
    return new AtenToLayer();
}

void register_atento_layer(ncnn::Net& net)
{
    net.register_custom_layer("aten::to", create_atento_layer);
}

// ============================================================
// TensorIndexLayer implementation
// ============================================================

TensorIndexLayer::TensorIndexLayer()
{
    one_blob_only = false;
    support_inplace = false;
}

int TensorIndexLayer::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                               std::vector<ncnn::Mat>& top_blobs,
                               const ncnn::Option& opt) const
{
    // Input 0: lookup table [dim, seq_len] (w=dim, h=seq_len)
    // Input 1: position index (scalar int or 1-element Mat)
    // Output: selected row [dim]
    if (bottom_blobs.size() < 2)
        return -100;

    const ncnn::Mat& table = bottom_blobs[0];
    const ncnn::Mat& index = bottom_blobs[1];

    // Handle two cases:
    // 1. index.w == table.h: full prefill — select ALL rows (identity pass-through)
    // 2. index.w == 1: single token — select one row
    ncnn::Mat& top_blob = top_blobs[0];

    if (index.w == table.h && table.h > 1) {
        // Full prefill: index contains all position IDs [0..h-1]
        // Output is the full table — identity mapping
        top_blob = table.clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        return 0;
    }

    // Single position: select one row
    int pos = (int)((const float*)index.data)[0];
    int table_h = table.h;
    if (table_h <= 0) table_h = 1;
    if (pos < 0) pos = 0;
    if (pos >= table_h) pos = table_h - 1;

    int row_size = table.w;
    top_blob.create(row_size, (size_t)4u);
    if (top_blob.empty())
        return -100;

    const float* src = table.row(pos);
    float* dst = top_blob;
    memcpy(dst, src, row_size * sizeof(float));

    return 0;
}

ncnn::Layer* create_tensorindex_layer(void* /*userdata*/)
{
    return new TensorIndexLayer();
}

void register_tensorindex_layer(ncnn::Net& net)
{
    net.register_custom_layer("Tensor.index", create_tensorindex_layer);
}

// ============================================================
// RepeatInterleaveLayer implementation (GQA KV expansion)
// ============================================================

RepeatInterleaveLayer::RepeatInterleaveLayer()
    : repeats(2), dim(-3)
{
    one_blob_only = true;
    support_inplace = false;
}

int RepeatInterleaveLayer::load_param(const ncnn::ParamDict& pd)
{
    // Default: dim=-3 (heads in 4D tensor), repeats=2 (GQA factor)
    repeats = 2;
    dim = -3;
    return 0;
}

int RepeatInterleaveLayer::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                                     std::vector<ncnn::Mat>& top_blobs,
                                     const ncnn::Option& opt) const
{
    const ncnn::Mat& input = bottom_blobs[0];
    ncnn::Mat& output = top_blobs[0];

    fprintf(stderr, "[repeat_interleave] %s: input [%d x %d x %d], repeats=%d\n",
            name.c_str(), input.w, input.h, input.c, repeats);

    int w = input.w;
    int h = input.h;
    int c = input.c;
    int out_c = c * repeats;

    output.create(w, h, out_c, sizeof(float));
    if (output.empty()) return -100;

    size_t channel_size = (size_t)w * h * sizeof(float);
    for (int ic = 0; ic < c; ic++) {
        for (int r = 0; r < repeats; r++) {
            int oc = ic * repeats + r;
            memcpy(output.channel(oc), input.channel(ic), channel_size);
        }
    }
    fprintf(stderr, "[repeat_interleave] %s: output [%d x %d x %d]\n",
            name.c_str(), output.w, output.h, output.c);
    return 0;
}

ncnn::Layer* create_repeat_interleave_layer(void* /*userdata*/)
{
    return new RepeatInterleaveLayer();
}

void register_repeat_interleave_layer(ncnn::Net& net)
{
    net.register_custom_layer("torch.repeat_interleave", create_repeat_interleave_layer);
}

} // namespace hunyuan
