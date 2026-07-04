# HunYuanOCR ncnn C++ Inference

> Repo: https://github.com/infinity955/ncnn_ocr

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

```bash
git clone https://github.com/Tencent/ncnn.git
```

### 1.2 Python Environment

```bash
# Python 3.10+
pip install torch==2.12.1 numpy pillow
pip install "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"
pip install pnnx==20260526
```

| Package | Version |
|---------|---------|
| Python | 3.10.18 |
| torch | 2.12.1 |
| transformers | 4.57.1 (commit `82a06db`) |
| pnnx | 20260526 |
| numpy | 2.4.2 |
| pillow | 10.4.0 |

### 1.3 Download HunYuanOCR Model

From HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download tencent/HunyuanOCR --local-dir ../hunyuanocrmodel
```

Or ModelScope (China):

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('tencent/HunyuanOCR', local_dir='../hunyuanocrmodel')"
```

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

**ViT** (dynamic resolution via two sizes):

```bash
PYTHONUTF8=1 python export_vision_ncnn.py
# → ncnn/vision.ncnn.{param,bin}
#    inputshape = [1,3,?,?]f32,[1,1152,?,?]f32
```

**Decoder** (dynamic sequence length):

```bash
cd ncnn
pnnx ../ts_pnnx/decoder.pt \
  inputshape=[1,8,1024],[1,1,8,8],[1,8,64],[1,8,64] \
  inputshape2=[1,16,1024],[1,1,16,16],[1,16,64],[1,16,64] \
  fp16=0 optlevel=2
# → decoder.ncnn.{param,bin}
```

**Text Embed**:

```bash
pnnx ../ts_pnnx/text_embed.pt \
  inputshape=[1,8]i64 inputshape2=[1,64]i64 fp16=0 optlevel=2
# → text_embed.ncnn.{param,bin}
```

**LM Head**:

```bash
pnnx ../ts_pnnx/lm_head.pt \
  inputshape=[1,8,1024]f32 inputshape2=[1,64,1024]f32 fp16=0 optlevel=2
# → lm_head.ncnn.{param,bin}
```

> All four modules should have **0 custom layers**. If any `pnnx.Expression` / `aten::to` appear, the export went wrong.

### 2.5 Add KV Cache

```bash
PYTHONUTF8=1 python hunyuan_ocr_add_kvcache.py
```

Modifies `decoder.ncnn.param`: each SDPA gets `cache_k/cache_v` I/O and `7=1` flag.
(732→733 layers, 971→1067 blobs)

### 2.6 Generate Tokenizer Files

```bash
PYTHONUTF8=1 python hunyuan_ocr_tokenizer.py
# → vocab.txt (120818 lines, line number = token ID)
```

### 2.7 Copy Models to Project

```bash
cp ncnn/vision.ncnn.{param,bin} ../hunyuanocr_ncnn/models/
cp ncnn/text_embed.ncnn.{param,bin} ../hunyuanocr_ncnn/models/
cp ncnn/decoder.ncnn.{param,bin} ../hunyuanocr_ncnn/models/
cp ncnn/lm_head.ncnn.param ../hunyuanocr_ncnn/models/
cp ncnn/pos_embed.bin ../hunyuanocr_ncnn/models/
cp vocab.txt ../hunyuanocr_ncnn/
```

---

## Phase 3: Build

### 3.1 Build ncnn

```bash
cd ncnn && mkdir build && cd build
cmake .. && make -j
```

Windows (MSVC):
```powershell
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release -j
```

### 3.2 Build hunyuanocr_ncnn

```bash
cd hunyuanocr_ncnn && mkdir build && cd build
cmake .. -Dncnn_DIR=<ncnn_root>/build/install/lib/cmake/ncnn
make -j
```

Windows (MSVC):
```powershell
cmake -G "Visual Studio 17 2022" -A x64 .. -DUSE_OPENMP=OFF
cmake --build . --config Release -j
```

---

## Phase 4: Run

```bash
./hunyuanocr_ncnn --model . --image test.jpg
```

| Option | Description |
|--------|-------------|
| `--model <dir>` | Model directory containing `model.json` (default `.`) |
| `--image <path>` | Input image (required) |
| `--prompt <text>` | OCR prompt (default: "检测并识别图片中的文字。") |

Windows PowerShell Chinese output:
```powershell
chcp 65001
.\hunyuanocr_ncnn.exe --model . --image test.jpg
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

| Image | HF PyTorch | Our ncnn C++ |
|-------|-----------|-------------|
| testimg.jpg | `顺利上岸(233,395),(761,628)LAMAR...` | `顺利上岸(231,392),(764,632)LAMAR...` |
| testimg2.jpg | `<pos_16><pos_121>P<USATRAVELER...` | `<pos_16><pos_121>P<USATRAVELER...` |

Text identical. Coordinates 2-6 px off (known `size=` vs `scale_factor+0.1` trade-off).

---

## Notes

These two fixes are **not in futz12's original** — they are specific to this repo.

**1. `int(gh)`/`int(gw)` removed → file `scripts/hyocr_modules_modify.py` line 109.**

Original futz12 code had `reshape(B, -1, int(gh), int(gw))`. The `int()` bakes traced
tensor dims as constants, crashing dual-size pnnx export. We removed the `int()` casts:
```python
x = x.reshape(1, c, -1).permute(0, 2, 1)  # uses -1, not h2*(w2+1)
```

**2. `.clone()` before vision injection → file `hunyuan_ocr.cpp` lines ~360-370.**

ncnn `Mat` uses COW (copy-on-write). `memcpy(embeds.row(pos), ...)` bypasses COW,
writing into shared data. After the text_embed Extractor is destroyed, freed memory
causes segfault. We insert `.clone()` before modification:
```cpp
text_embeds = text_embeds.clone();  // ← this line
for (int i = 0; i < num_vision_tokens; i++) {
    memcpy(text_embeds.row(image_positions[i]), image_embeds.row(i),
           hidden_size_ * sizeof(float));
}
```
Same fix applies in `generate()` (around line ~530) for re-injection during generation.

---

## References

- Original: [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm)
- Tutorial: [ncnn discussions #6793](https://github.com/Tencent/ncnn/discussions/6793)
