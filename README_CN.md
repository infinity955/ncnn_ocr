# HunYuanOCR ncnn C++ 推理

> 仓库: https://github.com/infinity955/ncnn_ocr

> 模型: https://huggingface.co/infinity955/ncnn_ocr

腾讯 HunYuanOCR（1B OCR模型）纯 ncnn C++ 推理。0 自定义层，动态分辨率，KV cache。

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

将 ncnn 克隆到**项目同级目录**下：

```bash
cd .. && git clone https://github.com/Tencent/ncnn.git && cd ncnn_ocr
# 最终目录结构:  YourProject/
#               ├── ncnn/              ← ncnn 源码
#               ├── ncnn_ocr/          ← 本项目
#               └── hunyuanocrmodel/   ← 下一步下载的模型
```

### 1.2 搭建 Python 环境

```bash
# Python 3.10+
pip install torch==2.12.1 torchvision numpy pillow accelerate huggingface_hub
pip install "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"
pip install pnnx==20260526
```

| 包 | 版本 |
|-----|------|
| Python | 3.10.18 |
| torch | 2.12.1 |
| torchvision | 0.27.1 |
| transformers | 4.57.1 (commit `82a06db`) |
| pnnx | 20260526 |
| accelerate | 1.14.0 |
| numpy | 2.4.2 |
| pillow | 10.4.0 |

### 1.3 下载 HunYuanOCR 原始模型

从 HuggingFace：

```bash
huggingface-cli download tencent/HunyuanOCR --local-dir ../hunyuanocrmodel
```

或从 ModelScope（国内）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('tencent/HunyuanOCR', local_dir='../hunyuanocrmodel')"
```

> 脚本会自动从项目根目录推导 `../hunyuanocrmodel/`。若下载到其他位置，可设 `MODEL_PATH` 环境变量或直接修改脚本中的 `MODEL` 变量。

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
ts/pos_embed.bin        # 位置嵌入基坐标 (1152,128,128) float32
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

> 所有 pnnx CLI 命令请在 **`scripts/ncnn/`** 目录下运行，确保输出文件位置正确。

**ViT**（两种分辨率实现动态 H/W）：

```bash
PYTHONUTF8=1 python export_vision_ncnn.py
# → ncnn/vision_encoder.ncnn.{param,bin}  （之后需重命名为 vision.ncnn.*）
#    inputshape = [1,3,?,?]f32,[1,1152,?,?]f32
```

**Decoder**（两个序列长度实现动态 L）：

```bash
cd ncnn
pnnx ../ts_pnnx/decoder.pt \
  ncnnparam=decoder.ncnn.param ncnnbin=decoder.ncnn.bin \
  inputshape=[1,8,1024],[1,1,8,8],[1,8,64],[1,8,64] \
  inputshape2=[1,16,1024],[1,1,16,16],[1,16,64],[1,16,64] \
  fp16=0 optlevel=2
# → decoder.ncnn.{param,bin}
```

**Text Embed**：

```bash
pnnx ../ts_pnnx/text_embed.pt \
  ncnnparam=text_embed.ncnn.param ncnnbin=text_embed.ncnn.bin \
  inputshape=[1,8]i64 inputshape2=[1,64]i64 fp16=0 optlevel=2
# → text_embed.ncnn.{param,bin}
```

**LM Head**：

```bash
pnnx ../ts_pnnx/lm_head.pt \
  ncnnparam=lm_head.ncnn.param ncnnbin=lm_head.ncnn.bin \
  inputshape=[1,8,1024]f32 inputshape2=[1,64,1024]f32 fp16=0 optlevel=2
# → lm_head.ncnn.{param,bin}
```

> 四个模块均应有 **0 个自定义层**。出现 `pnnx.Expression`/`aten::to` 说明导出有问题。

### 2.5 添加 KV Cache

```bash
PYTHONUTF8=1 python hunyuan_ocr_add_kvcache.py ncnn/decoder.ncnn.param
```

修改 `decoder.ncnn.param`：每个 SDPA 增加 `cache_k/cache_v` 输入输出和 `7=1`。
（732→733 层，971→1067 blob）。同时会保存备份 `decoder.ncnn.param.nokv`。

### 2.6 生成 Tokenizer 文件

```bash
PYTHONUTF8=1 python hunyuan_ocr_tokenizer.py ..
# → ../vocab.txt（120818 行，行号 = token ID）
# → ../merges.txt（119758 BPE merge 对）
```

### 2.7 复制模型到项目

```bash
# 重命名 vision_encoder → vision
mv ncnn/vision_encoder.ncnn.param ncnn/vision.ncnn.param
mv ncnn/vision_encoder.ncnn.bin   ncnn/vision.ncnn.bin

# 复制到项目 models/（从 scripts/ 目录，.. = 项目根目录）
cp ncnn/vision.ncnn.{param,bin}   ../models/
cp ncnn/text_embed.ncnn.{param,bin} ../models/
cp ncnn/decoder.ncnn.{param,bin}  ../models/
cp ncnn/lm_head.ncnn.param        ../models/
cp ts/pos_embed.bin               ../models/
cp vocab.txt merges.txt           ../
```

---

## 阶段三：编译

### 3.1 编译 ncnn

```bash
cd ../ncnn && mkdir -p build && cd build
cmake -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF .. && make -j4
cmake --install . --config Release   # 安装到 build/install/
```

Windows (MSVC)：
```powershell
cd ..\ncnn && mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release -j
cmake --install . --config Release
```

### 3.2 编译 ncnn_ocr

```bash
cd ../ncnn_ocr && mkdir -p build && cd build
cmake .. -Dncnn_DIR=$(realpath ../ncnn/build/install/lib/cmake/ncnn)
make -j4
```

Windows (MSVC)：
```powershell
cd ..\ncnn_ocr && mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 .. -Dncnn_DIR=../ncnn/build/install/lib/cmake/ncnn -DUSE_OPENMP=OFF
cmake --build . --config Release -j
```

---

## 阶段四：运行

```bash
# 在项目根目录 (ncnn_ocr/) 运行
./build/Release/hunyuanocr_ncnn --model . --image assets/testimg.jpg
```

| 参数 | 说明 |
|------|------|
| `--model <dir>` | 模型目录，包含 `model.json`（默认 `.`） |
| `--image <path>` | 输入图像（必填） |
| `--prompt <text>` | OCR 提示词（默认："检测并识别图片中的文字。"） |

Windows PowerShell 中文乱码：
```powershell
chcp 65001
.\build\Release\hunyuanocr_ncnn.exe --model . --image assets\testimg.jpg
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

**Python verify_modules.py**（逐模块 vs HF，testimg.jpg）：

| 模块 | maxabs | 状态 |
|------|--------|------|
| Vision Encoder | 2.815 | ✅ shape 匹配 |
| Text Embed | 0.000 | ✅ 完美 |
| Decoder | 3.671 | ✅ |
| LM Head | 1.588 | ✅ |
| **首 token argmax** | **120130 = 120130** | ✅ 一致 |

**C++ ncnn 推理**（端到端）：

| 图片 | 输出 |
|------|------|
| testimg.jpg | `顺利上岸(231,392),(764,632)LAMAR(446,850),(559,878)05483(934,835),(997,870)` |

文字与 PyTorch HF 完全一致。

---

## 注意事项 — 相对 futz12 原始脚本的改动

基于 futz12 原始脚本（见 `ncnn_llm/scripts/`）。主要改动：

### 1. 外部 pos_embed (`hyocr_modules_modify.py`)

原始 `hyocr_modules.py` 将 `pos_base` buffer（`1,1152,128,128`，~75MB）存储在 VisionEncoder
内部并自行插值。我们将其移到**模块外部**：
```python
# 原始: VisionEncoder.forward(self, pixels)
#   pos = F.interpolate(self.pos_base, size=[gh, gw], ...)
#   x = x + pos

# 我们的 modify 版: VisionEncoder.forward(self, pixels, pos_embed)
#   x = x + pos_embed  # 由调用方预插值后传入
```
这使得 `vision_encoder.pt` 减小约 75MB，且允许 C++ 驱动按每张图片的尺寸插值 `pos_embed`，
精确匹配 HF 预处理。

### 2. pos_embed 保存/加载流程

- `export_modules.py`: 从 HF 提取 pos_embed → 保存为 `pos_embed.bin`（替代复制到内部 buffer）
- `verify_modules.py`, `trace_export.py`, `export_vision_ncnn.py`: 加载 `pos_embed.bin`，插值到目标网格，传给 VisionEncoder

### 3. LayerNorm 复制修复 (`export_modules.py`)

原始代码 `cp(d.input_layernorm, s.input_layernorm)` 缺少 `bias=True`，导致 LayerNorm 的 bias
参数被跳过。修复为 `cp(d.input_layernorm, s.input_layernorm, bias=True)`。

### 4. 模型路径可配置

`MODEL` 从 `"tencent/HunyuanOCR"`（HF Hub 下载）改为可配置的本地路径，避免每次运行重复下载。

### 5. C++ COW 安全 (`hunyuan_ocr.cpp`)

ncnn `Mat` 使用写时复制。`memcpy(embeds.row(pos), ...)` 绕过 COW 写入共享数据，
Extractor 析构后释放可能导致 segfault。在修改前插入 `.clone()`（与 futz12 `ncnn_llm_ocr.cpp` 模式相同）：
```cpp
text_embeds = text_embeds.clone();
for (int i = 0; i < num_vision_tokens; i++)
    memcpy(text_embeds.row(image_positions[i]), image_embeds.row(i),
           hidden_size_ * sizeof(float));
```

---

## 参考

- 原始实现：[futz12/ncnn_llm](https://github.com/futz12/ncnn_llm)
- 完整教程：[ncnn discussions #6793](https://github.com/Tencent/ncnn/discussions/6793)
