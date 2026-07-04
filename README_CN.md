# HunYuanOCR ncnn C++ 推理

> 仓库: https://github.com/infinity955/ncnn_ocr

腾讯 HunYuanOCR（1B OCR 专家模型）纯 ncnn C++ 推理。0 自定义层，动态分辨率，KV cache。

文字输出与 PyTorch HF 完全一致，坐标差 2-6 px（见[注意事项](#注意事项)）。

---

## 快速开始

如果已有 ncnn 模型文件，跳到[阶段三](#阶段三编译)。

从 [Releases]() 下载预编译模型，解压到 `models/`：

```
models/
├── vision.ncnn.param / .bin       # ViT (~1.7 GB)
├── text_embed.ncnn.param / .bin   # Text Embed (~472 MB)
├── decoder.ncnn.param / .bin      # Decoder 带 KV cache (~1.6 GB)
├── lm_head.ncnn.param             # LM Head（bin 共享 text_embed）
└── pos_embed.bin                  # 位置嵌入权重 (~72 MB)
```

---

## 阶段一：环境搭建

### 1.1 克隆 ncnn 源码

```bash
git clone https://github.com/Tencent/ncnn.git
```

### 1.2 搭建 Python 环境

```bash
# Python 3.10+
pip install torch==2.12.1 numpy pillow
pip install "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"
pip install pnnx==20260526
```

| 包 | 版本 |
|-----|------|
| Python | 3.10.18 |
| torch | 2.12.1 |
| transformers | 4.57.1 (commit `82a06db`) |
| pnnx | 20260526 |
| numpy | 2.4.2 |
| pillow | 10.4.0 |

### 1.3 下载 HunYuanOCR 原始模型

从 HuggingFace：

```bash
pip install huggingface_hub
huggingface-cli download tencent/HunyuanOCR --local-dir ../hunyuanocrmodel
```

或从 ModelScope（国内）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('tencent/HunyuanOCR', local_dir='../hunyuanocrmodel')"
```

---

## 阶段二：模型导出（PyTorch → ncnn）

导出脚本在 `scripts/` 中。模型拆为 4 个独立模块，图文融合在 C++ 驱动代码完成。

### 2.1 导出 TorchScript

```bash
cd scripts
PYTHONUTF8=1 python export_modules.py
```

将 HF 权重加载到 `hyocr_modules_modify.py`（我们的修改版，外置 pos_embed）定义的干净模块中，用 `torch.jit.script` 导出。

```
ts/vision_encoder.pt    # ViT
ts/text_embed.pt        # Text Embed
ts/decoder.pt           # 24 层 Decoder
ts/lm_head.pt           # LM Head
```

### 2.2 精度验证（可选）

```bash
PYTHONUTF8=1 python verify_modules.py
```

预期：ViT maxabs ~2e-4，decoder ~3e-5，首个 token argmax 与 HF 一致。

### 2.3 转为 Traced

pnnx 需要 traced 图。从 scripted 的 state_dict 重建并 trace：

```bash
PYTHONUTF8=1 python trace_export.py
# → ts_pnnx/vision_encoder.pt, text_embed.pt, decoder.pt, lm_head.pt
```

### 2.4 pnnx → ncnn

**ViT**（两种分辨率实现动态 H/W）：

```bash
PYTHONUTF8=1 python export_vision_ncnn.py
# → ncnn/vision.ncnn.{param,bin}
#    inputshape = [1,3,?,?]f32,[1,1152,?,?]f32
```

**Decoder**（两个序列长度实现动态 L）：

```bash
cd ncnn
pnnx ../ts_pnnx/decoder.pt \
  inputshape=[1,8,1024],[1,1,8,8],[1,8,64],[1,8,64] \
  inputshape2=[1,16,1024],[1,1,16,16],[1,16,64],[1,16,64] \
  fp16=0 optlevel=2
# → decoder.ncnn.{param,bin}
```

**Text Embed**：

```bash
pnnx ../ts_pnnx/text_embed.pt \
  inputshape=[1,8]i64 inputshape2=[1,64]i64 fp16=0 optlevel=2
# → text_embed.ncnn.{param,bin}
```

**LM Head**：

```bash
pnnx ../ts_pnnx/lm_head.pt \
  inputshape=[1,8,1024]f32 inputshape2=[1,64,1024]f32 fp16=0 optlevel=2
# → lm_head.ncnn.{param,bin}
```

> 四个模块均应有 **0 个自定义层**。出现 `pnnx.Expression`/`aten::to` 说明导出有问题。

### 2.5 添加 KV Cache

```bash
PYTHONUTF8=1 python hunyuan_ocr_add_kvcache.py
```

修改 `decoder.ncnn.param`：每个 SDPA 增加 `cache_k/cache_v` 输入输出和 `7=1`。
（732→733 层，971→1067 blob）

### 2.6 生成 Tokenizer 文件

```bash
PYTHONUTF8=1 python hunyuan_ocr_tokenizer.py
# → vocab.txt（120818 行，行号 = token ID）
```

### 2.7 复制模型到项目

```bash
cp ncnn/vision.ncnn.{param,bin} ../hunyuanocr_ncnn/models/
cp ncnn/text_embed.ncnn.{param,bin} ../hunyuanocr_ncnn/models/
cp ncnn/decoder.ncnn.{param,bin} ../hunyuanocr_ncnn/models/
cp ncnn/lm_head.ncnn.param ../hunyuanocr_ncnn/models/
cp ncnn/pos_embed.bin ../hunyuanocr_ncnn/models/
cp vocab.txt ../hunyuanocr_ncnn/
```

---

## 阶段三：编译

### 3.1 编译 ncnn

```bash
cd ncnn && mkdir build && cd build
cmake .. && make -j
```

Windows (MSVC)：
```powershell
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release -j
```

### 3.2 编译 hunyuanocr_ncnn

```bash
cd hunyuanocr_ncnn && mkdir build && cd build
cmake .. -Dncnn_DIR=<ncnn_root>/build/install/lib/cmake/ncnn
make -j
```

Windows (MSVC)：
```powershell
cmake -G "Visual Studio 17 2022" -A x64 .. -DUSE_OPENMP=OFF
cmake --build . --config Release -j
```

---

## 阶段四：运行

```bash
./hunyuanocr_ncnn --model . --image test.jpg
```

| 参数 | 说明 |
|------|------|
| `--model <dir>` | 模型目录，包含 `model.json`（默认 `.`） |
| `--image <path>` | 输入图像（必填） |
| `--prompt <text>` | OCR 提示词（默认："检测并识别图片中的文字。"） |

Windows PowerShell 中文乱码：
```powershell
chcp 65001
.\hunyuanocr_ncnn.exe --model . --image test.jpg
```

---

## 架构

```
图像 → smart_resize + BICUBIC + BGR→RGB 归一化（CLIP mean/std）
     → ViT [1,3,H,W] + pos_embed [1,1152,gh,gw] → image_embeds (Lv,1024)

提示词 → BBPE 分词 → Text Embed → 注入 image_embeds
       → 合并 embeddings (L,1024)

Decoder（24 层，GQA 16/8，XD-RoPE，KV cache）→ hidden_states

LM Head（与 Text Embed 权重共享）→ logits (120818)
     → 贪心采样 → KV cache 增量推理直至 EOS (120007/120020)
```

### C++ 代码结构

| 文件 | 用途 |
|------|------|
| `main.cpp` | 入口，参数解析，Windows UTF-8 |
| `hunyuan_ocr.h/cpp` | `HunYuanOCR` 类：prefill + KV cache generate |
| `bpe_tokenizer.h/cpp` | Byte-level BPE tokenizer，含 byte decode |
| `rope_embed.h/cpp` | XD-RoPE 4 轴 cos/sin 生成 |
| `image_utils.h/cpp` | 图像加载、BICUBIC 缩放、smart_resize |
| `model.json` | 模型路径、token ID、RoPE 参数、vision 参数 |
| `vocab.txt` | 120818 行 token 词汇表（行号=ID） |

---

## 精度

| 图片 | HF PyTorch | ncnn C++ |
|------|-----------|----------|
| testimg.jpg | `顺利上岸(233,395),(761,628)LAMAR...` | `顺利上岸(231,392),(764,632)LAMAR...` |
| testimg2.jpg | `<pos_16><pos_121>P<USATRAVELER...` | `<pos_16><pos_121>P<USATRAVELER...` |

文字完全一致。坐标差 2-6 px（`size=` vs `scale_factor+0.1` 已知取舍）。

---

## 注意事项

以下两处是本仓库与 futz12 原版的**差异点**，已在代码中修复。

**1. `int(gh)`/`int(gw)` 已移除 → 文件 `scripts/hyocr_modules_modify.py` 第 109 行。**

原始代码 `reshape(B, -1, int(gh), int(gw))` 中的 `int()` 会将 traced tensor 转为常量，双尺寸 pnnx 导出时崩溃。本仓库已去掉 `int()`：
```python
x = x.reshape(1, c, -1).permute(0, 2, 1)  # 用 -1，不用 h2*(w2+1)
```

**2. Vision 注入前 `.clone()` → 文件 `hunyuan_ocr.cpp` 约 360 行。**

ncnn `Mat` 使用写时复制（COW）。`memcpy(embeds.row(pos), ...)` 绕过 COW 写入共享数据，text_embed Extractor 析构后数据释放导致 segfault。本仓库在修改前插入 `.clone()`：
```cpp
text_embeds = text_embeds.clone();  // ← 这一行
for (int i = 0; i < num_vision_tokens; i++) {
    memcpy(text_embeds.row(image_positions[i]), image_embeds.row(i),
           hidden_size_ * sizeof(float));
}
```
`generate()` 函数中（约 530 行）的 re-injection 有同样的 `.clone()` 处理。

---

## 参考

- 原始实现：[futz12/ncnn_llm](https://github.com/futz12/ncnn_llm)
- 完整教程：[ncnn discussions #6793](https://github.com/Tencent/ncnn/discussions/6793)
