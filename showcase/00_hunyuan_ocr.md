# HunYuanOCR → ncnn 纯 C++ 推理

> 将腾讯 HunYuanOCR（1B 多模态端到端 OCR）从 PyTorch 转换为 ncnn 格式，纯 C++ 推理，
> 输出与 PyTorch HF 逐字节一致。**0 自定义 ncnn 层**，动态分辨率，KV cache 增量解码。

## 最终结果

```
ncnn_ocr.exe --model . --image assets/testimg.jpg
→ 顺利上岸(231,392),(764,632)LAMAR(446,850),(559,878)05483(934,835),(997,870)
```

文字和坐标均与 PyTorch HF 完全一致，与其他两张标答图逐字验证通过。

### 三图验证结果

| 图片 | 输出 |
|------|------|
| testimg.jpg | `顺利上岸(231,392),(764,632)LAMAR(446,850),(559,878)05483(934,835),(997,870)` |
| testimg2.jpg | `<pos_16><pos_121><pos_981><pos_321>P<USATRAVELER<<HAPPY...` |
| testimg3.jpg | `搭子(139,586),(874,986)` |

## 获取模型

```bash
huggingface-cli download tencent/HunyuanOCR --local-dir ../hunyuanocrmodel
# 或魔搭:
python -c "from modelscope import snapshot_download; snapshot_download('tencent/HunyuanOCR', local_dir='../hunyuanocrmodel')"
```

模型约 2GB（bfloat16），架构为 `HunYuanVLForConditionalGeneration` = ViT 视觉编码器(27L) + LLM 解码器(24L)。

## 架构要点

| 部件 | 参数 |
|------|------|
| **ViT** | 27 层, hidden=1152, patch=16, spatial_merge=2 |
| **LLM 解码器** | 24 层, hidden=1024, GQA 16/8, head_dim=128, **XD-RoPE**(xdrope_section=[16,16,16,16]) |
| **LM Head** | Linear(1024→120818), tied with TextEmbed |
| **Tokenizer** | BBPE (GPT-2 byte-level), vocab 120818, special tokens 120000+ |

## PyTorch → ncnn 转换

### 路径选择：TorchScript 直接走 pnnx

ONNX 中转路径复杂且引入 op 兼容问题。采用 **`torch.jit.trace` → pnnx → ncnn**，利用 pnnx 对 TorchScript 的深度支持，
绕开 ONNX 中间格式。拆分为 4 个独立模块：

| 模块 | 输入 | 输出 |
|------|------|------|
| Vision Encoder | pixel_values + cos + sin (XD-RoPE) | image_embeds |
| Text Embed | token IDs | embeddings |
| Decoder | embeds + mask + cos + sin | hidden_states |
| LM Head | hidden_states | logits |

拆分的好处：图文融合在 C++ 端处理，每个模块独立 trace、独立 pnnx 转换、独立验证，问题隔离。

### 6 个 trace/导出级 Bug 修复

模型代码按 PyTorch eager 执行写的，`torch.jit.trace` 对动态特性敏感，需修复以下问题才能使 trace 通过：

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | ViT embeddings | `.item()` 在 trace 图中产生 Python 原语 | 去掉 `.item()` |
| 2 | PatchMerger | 同上，int 调 `.item()` | 同上 |
| 3 | ViT forward | 同上 | 同上 |
| 4 | generate() | dtype 不匹配（bf16 vs fp32） | 改用 `self.dtype` |
| 5 | XD-RoPE apply | cos/sin 维度不匹配（期望 4D 但拿到 2D） | 处理 2D cos |
| 6 | position_ids | 3D position_ids 未处理 | 加 `dim()==3` 分支 |

这些修复全部在本地 transformers forks 的 `modeling_hunyuan_vl.py` 中，不改模型数学行为。

### 位置编码外置 —— 坐标精度的关键

HF ViT 内部用 `F.interpolate(pos_base, size=[gh,gw])` 插值位置编码。
如果 C++ 端插入后直接用此方式，会导致 ViT maxabs ~2.82，最终坐标偏差 2-6 px。

**解决方案**：将 `pos_embed`(1152×128×128) **从模型内部移出**，保存为 `pos_embed.bin`（~72MB）。
C++ 驱动在运行时按每张图的网格、用与 HF **完全相同**的 `scale_factor=(h+0.1)/128` 双线性插值后再喂给 ViT。

```python
# 模型改造：pos_embed 改为外部输入
# 原: VisionEncoder.forward(self, pixels) → 内部 interpolate pos_embed
# 改: VisionEncoder.forward(self, pixels, pos_embed) → x = x + pos_embed  # 外部预插值
```

效果：ViT 精度从 2.82 降至 **0.0002**（等同 HF）→ 文字逐字一致，**坐标像素级一致**（2-6 px 偏差消失）。

### 4 模块全部 0 自定义层

经过 pnnx 转换后，四个 ncnn 模块（vision / text_embed / decoder / lm_head）**0 自定义层**，全部原生 ncnn 操作。
动态分辨率通过 DA-V2（两组输入尺寸的 dual-input）实现：
- Vision：`inputshape=[1,3,?,?]f32,[1,1152,?,?]f32`
- Decoder：`inputshape2=[1,16,1024],…`（两组 seq 长度）

## C++ 推理侧关键设计

### KV cache 增量解码

对 pnnx 生成的 SDPA 层手动改写：4 输入(1 输出) → 6 输入(3 输出),增加 `cache_k{i}/cache_v{i}` 输入和
`out_cache_k{i}/out_cache_v{i}` 输出。Prefill 时一次性提取 24 层 KV cache；generate 时单 token 输入，
复用已有 cache，逐 token 追加。

### XD-RoPE 4 轴位置编码

HunYuanOCR 使用 4 维位置编码(线性 + 高度 + 宽度 + 时间)：
- 文本 token 走 1D 标准 RoPE
- 图像 token 按空间坐标 (w, h, 0, 0) 走 4 轴 XD-RoPE
- C++ `generate_hunyuan_xdrope_cos_sin` 实现，含 NTK-alpha 基频扩展

### BBPE Tokenizer

GPT-2 风格的字节级 BPE tokenizer，含 byte-decoder（将字节 token 还原为 UTF-8）。
120818 vocab，特殊 token 从 120000 起。

### 逐模块验证

| 模块 | 方法 | 精度 |
|------|------|:--:|
| Vision Encoder | 外置 pos_embed 插值 | **0.0002** vs HF |
| Text Embed | 直接对比 | **0.000**（bit-exact） |
| Decoder | 24 SDPA + KV cache | **3.67** (maxabs) |
| LM Head | tied embedding | **1.59** (maxabs) |
| 首 token argmax | **120130 = 120130** ✅ |

## 构建与运行

```
cmake -S src -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j
build/Release/ncnn_ocr.exe --model . --image assets/testimg.jpg
```

---

*本项目参考了 ncnn_llm 仓库的 GLM-OCR 推理架构（KV cache 管理、文本运行时、模块拆分思路），在此致谢。*
