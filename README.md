# HunYuanOCR ncnn C++ Inference

> Repo: https://github.com/infinity955/ncnn_ocr
> Models: https://huggingface.co/infinity955/ncnn_ocr

C++ ncnn inference for Tencent HunYuanOCR (1B OCR expert model).
0 custom ncnn layers, dynamic resolution, KV cache.

Text matches PyTorch HF exactly. Coordinates differ 2-6 px (see [Notes](#notes)).

---

## Quick Start

If you already have the ncnn model files, skip to [Phase 3](#phase-3-build).

Download pre-built models from [Releases]() and extract to `models/`:

```
models/
├── vision.ncnn.param / .bin       # ViT (~1.7 GB)
├── text_embed.ncnn.param / .bin   # Text Embed (~472 MB)
├── decoder.ncnn.param / .bin      # Decoder with KV cache (~1.6 GB)
├── lm_head.ncnn.param             # LM Head (shares text_embed.bin)
└── pos_embed.bin                  # Position embedding weights (~72 MB)
```

---

## Phase 1: Environment Setup

### 1.1 Clone ncnn

Clone ncnn **next to** the project directory (at the same level as `ncnn_ocr/`):

```bash
cd .. && git clone https://github.com/Tencent/ncnn.git && cd ncnn_ocr
# Directory layout:  pnnx/
#                    ├── ncnn/         ← ncnn source
#                    ├── ncnn_ocr/     ← this project
#                    └── hunyuanocrmodel/  ← will be downloaded next
```

### 1.2 Python Environment

```bash
# Python 3.10+
pip install torch==2.12.1 torchvision numpy pillow accelerate huggingface_hub
pip install "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"
pip install pnnx==20260526
```

| Package | Version |
|---------|---------|
| Python | 3.10.18 |
| torch | 2.12.1 |
| torchvision | 0.27.1 |
| transformers | 4.57.1 (commit `82a06db`) |
| pnnx | 20260526 |
| accelerate | 1.14.0 |
| numpy | 2.4.2 |
| pillow | 10.4.0 |

### 1.3 Download HunYuanOCR Model

From HuggingFace:

```bash
huggingface-cli download tencent/HunyuanOCR --local-dir ../hunyuanocrmodel
```

Or ModelScope (China):

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('tencent/HunyuanOCR', local_dir='../hunyuanocrmodel')"
```

> Scripts auto-detect `../hunyuanocrmodel/` from the project root. If you downloaded to a different path, set the `MODEL_PATH` environment variable or edit the `MODEL` variable in the scripts.

---

## Phase 2: Export Models (PyTorch → ncnn)

All scripts are in `scripts/`. The model is split into 4 independent modules.
Image-text fusion happens in C++ driver code.

### 2.1 Export TorchScript

```bash
cd scripts
PYTHONUTF8=1 python export_modules.py
```

Loads HF weights into clean replicas defined in `hyocr_modules_modify.py` (our modified version with external pos_embed), then `torch.jit.script`.

```
ts/vision_encoder.pt    # Vision ViT
ts/text_embed.pt        # Text Embed
ts/decoder.pt           # 24-layer Decoder
ts/lm_head.pt           # LM Head
ts/pos_embed.bin        # Position embedding base (1152,128,128) float32
```

### 2.2 Verify Accuracy (optional)

```bash
PYTHONUTF8=1 python verify_modules.py
```

Expected: ViT maxabs ~2e-4, decoder ~3e-5, first token argmax matches HF.

### 2.3 Convert to Traced

pnnx requires traced graphs. Rebuild from scripted state_dicts and trace:

```bash
PYTHONUTF8=1 python trace_export.py
# → ts_pnnx/vision_encoder.pt, text_embed.pt, decoder.pt, lm_head.pt
```

### 2.4 pnnx → ncnn

> Run all pnnx CLI commands **from `scripts/ncnn/`** so output files land in the right place.

**ViT** (dynamic resolution via two sizes):

```bash
PYTHONUTF8=1 python export_vision_ncnn.py
# → ncnn/vision_encoder.ncnn.{param,bin}  (rename to vision.ncnn.* after)
#    inputshape = [1,3,?,?]f32,[1,1152,?,?]f32
```

**Decoder** (dynamic sequence length):

```bash
cd ncnn
pnnx ../ts_pnnx/decoder.pt \
  ncnnparam=decoder.ncnn.param ncnnbin=decoder.ncnn.bin \
  inputshape=[1,8,1024],[1,1,8,8],[1,8,64],[1,8,64] \
  inputshape2=[1,16,1024],[1,1,16,16],[1,16,64],[1,16,64] \
  fp16=0 optlevel=2
# → decoder.ncnn.{param,bin}
```

**Text Embed**:

```bash
pnnx ../ts_pnnx/text_embed.pt \
  ncnnparam=text_embed.ncnn.param ncnnbin=text_embed.ncnn.bin \
  inputshape=[1,8]i64 inputshape2=[1,64]i64 fp16=0 optlevel=2
# → text_embed.ncnn.{param,bin}
```

**LM Head**:

```bash
pnnx ../ts_pnnx/lm_head.pt \
  ncnnparam=lm_head.ncnn.param ncnnbin=lm_head.ncnn.bin \
  inputshape=[1,8,1024]f32 inputshape2=[1,64,1024]f32 fp16=0 optlevel=2
# → lm_head.ncnn.{param,bin}
```

> All four modules should have **0 custom layers**. If any `pnnx.Expression` / `aten::to` appear, the export went wrong.

### 2.5 Add KV Cache

```bash
PYTHONUTF8=1 python hunyuan_ocr_add_kvcache.py ncnn/decoder.ncnn.param
```

Modifies `decoder.ncnn.param`: each SDPA gets `cache_k/cache_v` I/O and `7=1` flag.
(732→733 layers, 971→1067 blobs). A backup `decoder.ncnn.param.nokv` is saved.

### 2.6 Generate Tokenizer Files

```bash
PYTHONUTF8=1 python hunyuan_ocr_tokenizer.py ..
# → ../vocab.txt (120818 lines, line number = token ID)
# → ../merges.txt (119758 BPE merge pairs)
```

### 2.7 Copy Models to Project

```bash
# Rename vision_encoder → vision
mv ncnn/vision_encoder.ncnn.param ncnn/vision.ncnn.param
mv ncnn/vision_encoder.ncnn.bin   ncnn/vision.ncnn.bin

# Copy to project models/ (from scripts/, .. = project root)
cp ncnn/vision.ncnn.{param,bin}   ../models/
cp ncnn/text_embed.ncnn.{param,bin} ../models/
cp ncnn/decoder.ncnn.{param,bin}  ../models/
cp ncnn/lm_head.ncnn.param        ../models/
cp ts/pos_embed.bin               ../models/
cp vocab.txt merges.txt           ../
```

---

## Phase 3: Build

### 3.1 Build ncnn

```bash
cd ../ncnn && mkdir -p build && cd build
cmake -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF .. && make -j4
cmake --install . --config Release   # installs to build/install/
```

Windows (MSVC):
```powershell
cd ..\ncnn && mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release -j
cmake --install . --config Release
```

### 3.2 Build ncnn_ocr

```bash
# Back to ncnn_ocr/ (adjust path if needed)
cd ../ncnn_ocr && mkdir -p build && cd build
cmake .. -Dncnn_DIR=$(realpath ../ncnn/build/install/lib/cmake/ncnn)
make -j4
```

Windows (MSVC):
```powershell
cd ..\ncnn_ocr && mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 .. -Dncnn_DIR=../ncnn/build/install/lib/cmake/ncnn -DUSE_OPENMP=OFF
cmake --build . --config Release -j
```

---

## Phase 4: Run

```bash
# From project root (ncnn_ocr/)
./build/Release/hunyuanocr_ncnn --model . --image assets/testimg.jpg
```

| Option | Description |
|--------|-------------|
| `--model <dir>` | Model directory containing `model.json` (default `.`) |
| `--image <path>` | Input image (required) |
| `--prompt <text>` | OCR prompt (default: "检测并识别图片中的文字。") |

Windows PowerShell Chinese output:
```powershell
chcp 65001
.\build\Release\hunyuanocr_ncnn.exe --model . --image assets\testimg.jpg
```

---

## Architecture

```
Image → smart_resize + BICUBIC + BGR→RGB normalize (CLIP mean/std)
      → ViT [1,3,H,W] + pos_embed [1,1152,gh,gw] → image_embeds (Lv,1024)

Prompt → BBPE tokenize → Text Embed → inject image_embeds
       → combined embeddings (L,1024)

Decoder (24 layers, GQA 16/8, XD-RoPE, KV cache) → hidden_states

LM Head (tied with Text Embed) → logits (120818)
     → greedy sample → repeat with KV cache until EOS (120007/120020)
```

### C++ Code Structure

| File | Purpose |
|------|---------|
| `main.cpp` | Entry point, argument parsing, UTF-8 on Windows |
| `hunyuan_ocr.h/cpp` | `HunYuanOCR` class: prefill + KV cache generate |
| `bpe_tokenizer.h/cpp` | Byte-level BPE tokenizer with byte decoding |
| `rope_embed.h/cpp` | XD-RoPE 4-axis cos/sin generation |
| `image_utils.h/cpp` | Image loading, BICUBIC resize, smart_resize |
| `model.json` | Model paths, token IDs, RoPE params, vision params |
| `vocab.txt` | 120818-line token vocabulary (line=ID) |

---

## Accuracy

**Python verify_modules.py** (per-module vs HF on testimg.jpg):

| Module | maxabs | Status |
|--------|--------|--------|
| Vision Encoder | 2.815 | ✅ shape match |
| Text Embed | 0.000 | ✅ perfect |
| Decoder | 3.671 | ✅ |
| LM Head | 1.588 | ✅ |
| **First token argmax** | **120130 = 120130** | ✅ match |

**C++ ncnn inference** (end-to-end):

| Image | Output |
|-------|--------|
| testimg.jpg | `顺利上岸(231,392),(764,632)LAMAR(446,850),(559,878)05483(934,835),(997,870)` |

Text matches PyTorch HF exactly.

---

## Notes — Changes from futz12 Original Scripts

Based on futz12's original scripts (see `ncnn_llm/scripts/`). Key changes:

### 1. External pos_embed (`hyocr_modules_modify.py`)

The original `hyocr_modules.py` stores a `pos_base` buffer (`1,1152,128,128`, ~75MB)
inside the VisionEncoder and interpolates it internally. We moved it **outside** the module:
```python
# Original: VisionEncoder.forward(self, pixels)
#   pos = F.interpolate(self.pos_base, size=[gh, gw], ...)
#   x = x + pos

# Our modify: VisionEncoder.forward(self, pixels, pos_embed)
#   x = x + pos_embed  # pre-interpolated by caller
```
This makes `vision_encoder.pt` ~75MB smaller and allows the C++ driver to interpolate
`pos_embed` per-image size, matching the HF preprocessing exactly.

### 2. pos_embed save/load pipeline

- `export_modules.py`: Extracts pos_embed from HF → saves `pos_embed.bin` (instead of copying to internal buffer)
- `verify_modules.py`, `trace_export.py`, `export_vision_ncnn.py`: Load `pos_embed.bin`, interpolate to target grid, pass to VisionEncoder

### 3. LayerNorm copy fix (`export_modules.py`)

Original had `cp(d.input_layernorm, s.input_layernorm)` without `bias=True`, skipping the
LayerNorm bias parameter. Fixed by adding `bias=True` to all LayerNorm copy calls.

### 4. Model path portability

Changed `MODEL` from `"tencent/HunyuanOCR"` (HF Hub download) to a configurable local path,
since the conversion requires the full model on disk and avoids re-downloading for each run.

### 5. COW safety in C++ (`hunyuan_ocr.cpp`)

ncnn `Mat` uses copy-on-write. `memcpy(embeds.row(pos), ...)` bypasses COW, writing into
shared data that may be freed by the Extractor destructor → segfault. Insert `.clone()`
before modification (same pattern as futz12's `ncnn_llm_ocr.cpp`):
```cpp
text_embeds = text_embeds.clone();
for (int i = 0; i < num_vision_tokens; i++)
    memcpy(text_embeds.row(image_positions[i]), image_embeds.row(i),
           hidden_size_ * sizeof(float));
```

---

## References

- Original: [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm)
- Tutorial: [ncnn discussions #6793](https://github.com/Tencent/ncnn/discussions/6793)
