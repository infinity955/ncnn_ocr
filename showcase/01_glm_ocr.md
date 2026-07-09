# GLM-OCR → ncnn 纯 C++ 推理

> 将 `zai-org/GLM-OCR`（1.11B 多模态 OCR）**由自己的管线**从 PyTorch 转为 ncnn，纯 C++ 推理，
> 输出与标答一致。**0 自定义 ncnn 层**，**0 复用预转换模型**。承接 HunYuanOCR 的转换经验，
> 在分层验证、ncnn 友好写法、KV cache 改写等方面直接复用已验证模式。

## 最终结果

```
ncnn_ocr.exe --model assets/glm_ocr --image assets/testimg.jpg --prompt "识别图片中的文字"
→ 顺利上岸  LAMAR  054839
```

与 PyTorch HF 输出一致，正确停机。

## 获取模型

```bash
python scripts/glm/download.py --output ../glm_ocr_model
# 或: modelscope download --model ZhipuAI/glm-ocr --local_dir ../glm_ocr_model
```

模型 1828 MB（float32），架构为 `GlmOcrForConditionalGeneration` = ViT 视觉编码器(24L) + LLM 解码器(16L)。
transformers 需 **≥5.13.0**（PyPI 正式版已内置 `glm_ocr`），与 HunYuan 的定制 fork 互斥，用独立 venv。

## 架构要点（与 HunYuan 的关键差异）

| 方面 | HunYuanOCR | GLM-OCR |
|------|-----------|---------|
| Vision patch | Conv2d | **Conv3d**(temporal_patch=2) |
| Vision 输出 | proj Linear | **门控 merger** + downsample Conv2d |
| Decoder norm | 标准 2-norm | **sandwich 4-norm**（每层 4 个 RMSNorm） |
| Text MLP | 分离 gate/up | **融合 gate_up_proj**（单 Linear→chunk） |
| Text RoPE | XD-RoPE 4 轴 | **mRoPE 3 轴**(mrope_section=[16,24,24]) |
| Vision RoPE | 无（绝对位置编码） | **2D mRoPE**（[16,16]，宽 32→64） |
| Text attn | 有 q/k norm | **无 q/k norm** |
| GQA | 16/8 | 16/8 |
| Tokenizer | BBPE(120818) | BBPE(59392) |
| 特殊 token | 120000+ | 59246+(added_tokens 36 个) |

## 从 HunYuan 继承的转换基础设施

GLM 的转换管线直接复用了 HunYuan 已验证的经验：

- **分层逐模块对齐**：clean 模块 vs HF → ncnn vs 模块 → C++ 端到端，每层 pass 再往下
- **pnnx 转换模式**：`torch.jit.trace`(非 script) → `pnnx.export`(DA-V2 动态分辨率)
- **ncnn 友好写法**：3D cos/sin 广播（非 4D unsqueeze）、`.contiguous()`、避免 `expand`→`Expression`
- **KV cache 改写**：HunYuan 的 `add_kvcache.py` 格式（header 在 2 行、SDPA 按顺序编号）直接复用
- **RMSNorm 0 自定义层**：去掉 dtype 转换（fp32 无需）

## 新遇到的 5 个核心技术难点

### 1. Conv3d 时序 patch 折叠

GLM 的 vision patch embed 是 `Conv3d(3,1024,kernel=[2,14,14])`，`temporal_patch_size=2`。
processor 把单图沿时序复制成 2 帧（两帧完全相同）。

**解法**：`W_conv2d = W_conv3d.sum(dim=temporal)`，bias 不变。因为 Conv3d 对两帧相同输入求和 = Conv2d 对单帧。
C++ 侧只喂单帧 strip，等价成立，无需处理 3D 卷积。验证 maxabs **1.9e-6** vs HF。

### 2. 交错 RoPE → 权重置换转连续 RoPE（GLM 独有难点）

GLM text 用**交错** RoPE（`x[0::2]/x[1::2]` + `repeat_interleave(2)`），pnnx 转成 step-2 slice，
ncnn 直接报 `slice with step 2 is not supported`（64 个 SDPA 层全炸）。

**解法**：在导出权重时**置换 q/k 投影的每头行序**（交错→前后半：`[0,2,4,…,1,3,5,…]`），然后模块改为
**连续** RoPE（`cat([cos,cos])` + 前后半 `rotate_half`）。

```python
# 置换每头 head_dim 内的行序：偶数在前半，奇数在后半
def perm_interleave_to_half(w, n_heads, head_dim):
    W = w.view(n_heads, head_dim, -1)
    idx = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
    return W[:, idx, :].reshape(-1, w.shape[1])
```

**数学等价性**：`Q·Kᵀ` 对 Q 和 K 施同一行列置换不变。V 和 o_proj 不动。所以置换后模块改用连续
RoPE，pnnx 识别为原生 `RotaryEmbed`（32 个），**0 非原生操作**。验证 decoder maxabs **2.86e-6**
vs HF（置换前后）。

> HunYuan/PaddleOCR-VL 天然是连续 RoPE，不需要这一步——这是 GLM 独有的额外工作量。

### 3. Sandwich 4-norm + 融合 gate_up_proj

GLM 的 decoder layer 每层 4 个 RMSNorm（HunYuan 只有 2 个）：

```
r = h;  h = input_ln(h);        h = attn(h);  h = post_self_attn_ln(h);   h = r + h
r = h;  h = post_attention_ln(h); h = mlp(h);  h = post_mlp_ln(h);        h = r + h
```

MLP 是**融合** `gate_up_proj`(1536→9216)→`chunk(2)`→`SiLU(gate)*up`→`down`。
text attention **没有 q/k norm**（只有 vision attn 有）。

### 4. ncnn 布局不兼容：batch 轴广播 + Conv2d downsample 塌缩

两个问题都在 HunYuan 上已解决，GLM 直接套用：

- **RoPE 4D 广播**（`cos.unsqueeze(2)` → ncnn `batch axis 233 not supported`）：改用 3D cos/sin `(1,n,dim)`，靠右对齐自动广播过 heads
- **Conv2d downsample 塌缩**（`(N/4,C,2,2)` 被 ncnn 把 N/4 当通道）：换成 `Linear(hidden*4, out)`，保持 `(N/4, feature)` 行布局

### 5. tokenizer 特殊 token 补全 + bbpe 字节解码 + eos_ids

GLM 的 36 个特殊 token（`<|image|>`、`<|begin_of_image|>` 等）独立存放在 `added_tokens`，不在 base vocab。
漏掉它们会导致 `<|image|>` 被 BPE 拆碎、图像不注入、输出乱码。
需按 ID 补进词表并 pad 到 59392。Token type 必须用 `bbpe` 触发 byte-decoder（`é¡ºåĪ©ä¸Ĭå²¸`→「顺利上岸」）。
停机 token 是集合 `eos_ids=[59246,59253]`（`<|endoftext|>` + `<|user|>`）。

## 分层验证

| 层 | 对比 | 结果 |
|----|------|:--:|
| clean 模块 vs HF | PyTorch↔PyTorch | embed/decoder/lmhead **0.0**，vision **1.9e-6** |
| ncnn vs PyTorch 模块 | ncnn↔PyTorch | text_embed **0.0**，lm_head 3e-6，decoder **9e-4**，vision 2e-4 |
| C++ vs 标答 | 端到端 | ✅ 顺利上岸 LAMAR 054839，正确停机 |

## 构建与运行

```
ncnn_ocr.exe --model assets/glm_ocr --image assets/testimg.jpg --prompt "识别图片中的文字"
```

---
