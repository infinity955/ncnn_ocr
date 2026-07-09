# PaddleOCR-VL-1.6 → ncnn 纯 C++ 推理

> 将 PaddleOCR-VL-1.6（0.9B VLM 文档解析器，OmniDocBench v1.6 **96.33% SOTA**，宣称超 GLM-OCR）
> **由自己的管线**从 PyTorch 转为 ncnn，纯 C++ 推理。**0 自定义 ncnn 层**，**0 复用预转换模型**。
> 三模型中唯一一个**greenfield C++ 实现**（前两个有 ncnn_llm 参考），也是三者中
> **解码器最简单**的（2-norm/分离 MLP/连续 RoPE），但**视觉+tokerizer 最特殊**。

## 最终结果

```
ncnn_ocr.exe --model assets/paddleocr_vl --image assets/testimg.jpg --prompt ""
→ 顺利上岸  LAMAR  05483
```

⚠️ **不需要 prompt**（空字符串即可）。PaddleOCR-VL 是文档解析器而非对话 VLM——给 prompt 反而不行。

## 获取模型

```bash
python scripts/paddleocr_vl/download.py --output ../paddleocr_vl_model
# 或: modelscope download --model PaddlePaddle/PaddleOCR-VL-1.6 --local_dir ../paddleocr_vl_model
```

模型 1828 MB（bfloat16），架构为 `PaddleOCRVLForConditionalGeneration` = SigLIP-so400m 视觉(27L)
+ Projector(2×2 merge) + ERNIE-4.5 解码器(18L)。需 `trust_remote_code=True`，remote code 针对
transformers **4.55.0** 编写。

## 与其他两个模型的关键差异

PaddleOCR-VL 和我们已做的 HunYuan/GLM 有本质区别：

| 方面 | HunYuan / GLM | PaddleOCR-VL |
|------|---------------|------|
| **定位** | 对话式 VLM（给 prompt 做 OCR） | **文档解析器**（给图片即输出文字+坐标） |
| 解码器 norm | sandwich 4-norm / 2-norm | **标准 2-norm pre-norm**（最简单） |
| MLP | 融合 gate_up(需 chunk) | **分离 gate/up/down**（直接拷） |
| 文本 RoPE | 交错(需权重置换) / XD-RoPE | **NEOX 连续**（天然 ncnn 友好） |
| 视觉 | ViT + 外置 pos_embed | **SigLIP**：学习式 pos_embed + **2D RoPE**（双份位置编码） |
| Tokenizer | GPT-2 BBPE | **SentencePiece BPE + byte_fallback**（`<0xHH>`→字节） |
| Prompt | 需要 prompt | **空字符串**即可 |
| C++ 实现 | 参考 ncnn_llm | **从零编写**(~310 行,复用共享工具) |

## 继承与复用

从 GLM/HunYuan 直接复用了已验证的转换基础设施:
- **分层验证框架**:clean 模块→ncnn→C++,每层 pass 再往下
- **ncnn 友好写法**:3D cos/sin 广播,RMSNorm 去 dtype 转换,避免 `expand`→`Expression`
- **pnnx 转换**:`torch.jit.trace` + DA-V2 动态分辨率
- **KV cache** (`add_kvcache.py`):HunYuan 的格式直接复用(18 SDPA→6/3,36 cache blob)
- **共享 C++ 工具**:`text_runtime` / `vision_rope` / `rope_embed` / `sampling` / `image_utils` / `bpe_tokenizer` 全部复用

解码器侧天然避免了 GLM 遇到的几个坑(连续 RoPE→免权重置换,分离 MLP→免 chunk,2-norm→少 2 个 RMSNorm)。

## 新遇到的 5 个核心技术难点

### 1. 视觉：同时有学习式 pos_embed + 2D RoPE（双份位置编码）

这是 PaddleOCR-VL 最特殊的设计。SigLIP-so400m(27L,LayerNorm,gelu_tanh)**同时使用两种位置编码**:
- **学习式** `position_embedding`(Embedding 729=27×27,1152) → 双线性插值到目标网格
- **2D RoPE**(连续 `rotate_half`,dim 36→72,theta 10000,H/W 两轴分块 [18,18])

两种编码**叠加使用**,缺一不可。C++ 实现需两条路径:

**① 外置 pos_embed + 双线性插值**(借 HunYuan 经验):
将 `position_embedding.weight`(27×27×1152) 存为 `pos_embed.bin`。C++ 运行时按实际网格做
`F.interpolate(bilinear, align_corners=False)` 后喂给 vision net。
**精度关键**:插值公式必须与 PyTorch 完全一致。已验证 **maxabs=0.00e+00** vs HF。

**② 2D RoPE**(借 GLM 的 `generate_vision_rope_cache_2d`):
用 `[18,18]` 分块(H/W 各 18 个频率项,和=36),block-major 顺序生成 cos/sin,喂给 vision net。

Vision 模块因此是 **4 输入**(strip + pos_embed + cos + sin),比其他模型多一个 blob。

### 2. SentencePiece + byte_fallback Tokenizer（全新解码路径）

PaddleOCR-VL 的 tokenizer **不是** GPT-2 BBPE,而是 SentencePiece 风格 BPE:

| 特性 | GLM/HunYuan (GPT-2 BBPE) | PaddleOCR-VL (SP BPE) |
|------|--------------------------|------------------------|
| 词边界 | 无(字节级) | `▁`(U+2581)→空格 |
| OOV | byte_encoder map | **byte_fallback**(`<0xHH>`→原始字节) |
| 解码 | `ByteDecode` | `▁`→空格 + `<0xHH>`→字节融合→UTF-8 |

**解决方案**:在共享 `BpeTokenizer::decode()` 的 SP 分支增加 byte_fallback 处理:
- 检测 `<0xHH>` 模式 → 转换为原始字节(相邻字节自动融合为 UTF-8 字符)
- vocab.txt 用 `\n`/`\r` 转义(部分 SP token 含字面换行,否则行号=ID 对不上)

### 3. 不需要 prompt —— 模型定位差异

PaddleOCR-VL 定位是**文档解析器**(image→text+coordinates),不是对话 VLM。
空字符串即可触发默认文档解析模式,直接输出文字+坐标。给 prompt 反而会让模型"思考"而非 OCR。

这导致了生成时需**过滤输出 token**:模型会输出 `<|LOC_*|>` 坐标标注 token,需要在 generate loop 识别并跳过,
只保留文字部分。

### 4. ERNIE-4.5 解码器的 GQA 16/2

与 GLM 的 GQA 16/8 不同,PaddleOCR-VL 使用极端的 **GQA 16/2**(`q_per_kv=8`),
对 ncnn 的 `repeat_interleave` 和 SDPA 的数值精度压力更大。
这也是 ncnn 级验证时 decoder hidden 偏差(4.9e-2)比 GLM(9e-4)大的原因——但 **argmax+top5 完全一致**,
贪心解码不受影响。

### 5. 外置学习式 pos_embed 的双线性插值精度

与 HunYuan 的固定尺寸 pos_embed(128×128→网格)不同,PaddleOCR-VL 的 base 是 27×27,
使用 `align_corners=False` 的 bilinear 插值。公式验证 `maxabs=0.00e+00` vs HF,
**C++ 实现必须逐像素对齐**——这是 C++ greenfield 中最容易出错的一步。

## 分层验证

| 层 | 对比 | 结果 |
|----|------|:--:|
| clean 模块 vs HF | PyTorch↔PyTorch | embed/decoder/lmhead **0.0**，vision **6.7e-6** |
| ncnn vs PyTorch 模块 | ncnn↔PyTorch | text_embed/lmhead 0.0, vision 2.8e-3 |
| C++ vision vs HF(真图) | C++↔PyTorch | pos_embed **0.0**, Projector **2.82e-4** |
| C++ prefill argmax | 端到端 | next_token **23=23**(匹配 HF) |
| C++ vs 标答 | 端到端 | ✅ 顺利上岸 LAMAR 05483 |

## 构建与运行

```
ncnn_ocr.exe --model assets/paddleocr_vl --image assets/testimg.jpg --prompt ""
```

## 三模型对比（一张图，一个 exe）

```
# HunYuanOCR
ncnn_ocr.exe --model . --image assets/testimg.jpg
→ 顺利上岸(231,392),(764,632)LAMAR(446,850),(559,878)05483(934,835),(997,870)

# GLM-OCR
ncnn_ocr.exe --model assets/glm_ocr --image assets/testimg.jpg --prompt "识别图片中的文字"
→ 顺利上岸  LAMAR  054839

# PaddleOCR-VL-1.6
ncnn_ocr.exe --model assets/paddleocr_vl --image assets/testimg.jpg --prompt ""
→ 顺利上岸  LAMAR  05483
```
