#!/usr/bin/env python3
"""
Convert the GLM-OCR clean modules to ncnn via pnnx.

Loads state_dicts from ts/ (saved by export_modules.py), rebuilds eager modules
(defaults already match the real glm_ocr config), traces, and runs pnnx.export
with DA-V2 (two input sizes) so the ncnn graph keeps dynamic patch-count / seq-len.

Output names match glm_model.json params: glm_{vision,text_embed,decoder,lm_head}.ncnn.*

Usage:
    .venv-glm/Scripts/python.exe scripts/glm/export_vision_ncnn.py --input ts --output ncnn
"""

import argparse
import os

import torch
import pnnx

from modules import VisionEncoder, TextEmbed, Decoder, LMHead, build_causal_mask

HERE = os.path.dirname(os.path.abspath(__file__))


def load(mod, ts_dir, name):
    mod.load_state_dict(torch.load(os.path.join(ts_dir, name + ".pt")))
    return mod.eval()


def export_one(mod, out_dir, out_name, inputs, inputs2=None):
    # pnnx embeds these paths into a generated *_pnnx.py as raw string literals;
    # backslashes like "\ncnn" become escape sequences -> use forward slashes.
    j = lambda *p: os.path.join(*p).replace("\\", "/")
    pt_path = j(out_dir, out_name + ".pt")
    torch.jit.trace(mod, inputs, check_trace=False).save(pt_path)
    kw = dict(inputs=inputs, optlevel=2, fp16=False, device="cpu",
              pnnxparam=j(out_dir, out_name + ".pnnx.param"),
              pnnxbin=j(out_dir, out_name + ".pnnx.bin"),
              pnnxpy=j(out_dir, out_name + "_pnnx.py"),
              pnnxonnx=j(out_dir, out_name + ".pnnx.onnx"),
              ncnnparam=j(out_dir, out_name + ".ncnn.param"),
              ncnnbin=j(out_dir, out_name + ".ncnn.bin"),
              ncnnpy=j(out_dir, out_name + "_ncnn.py"))
    if inputs2 is not None:
        kw["inputs2"] = inputs2
    try:
        pnnx.export(mod, pt_path, **kw)
    except Exception as e:
        # pnnx's final validation (running the generated _pnnx.py) can fail even
        # after the ncnn files are written correctly; tolerate if outputs exist.
        print(f"  (pnnx post-step: {type(e).__name__})")
    ok = os.path.exists(kw["ncnnparam"]) and os.path.exists(kw["ncnnbin"])
    print(f"  {out_name}: {'OK' if ok else 'MISSING OUTPUT'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default=os.path.join(HERE, "ts"))
    ap.add_argument("--output", "-o", default=os.path.join(HERE, "ncnn"))
    ap.add_argument("--only", default="", help="comma list: vision,text_embed,decoder,lm_head")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    want = lambda n: (not only) or (n in only)

    # ---- Vision (dynamic patch count N, multiple of 4) ----
    if want("vision"):
        print("=== Vision Encoder ===")
        vis = load(VisionEncoder(), args.input, "vision_encoder")
        N1, N2 = 64, 16
        i1 = (torch.randn(1, 3, 14, 14 * N1), torch.randn(1, N1, 32), torch.randn(1, N1, 32))
        i2 = (torch.randn(1, 3, 14, 14 * N2), torch.randn(1, N2, 32), torch.randn(1, N2, 32))
        export_one(vis, args.output, "glm_vision", i1, i2)

    # ---- Text Embed (dynamic seq len) ----
    if want("text_embed"):
        print("=== Text Embed ===")
        emb = load(TextEmbed(), args.input, "text_embed")
        i1 = torch.zeros(1, 16, dtype=torch.long)
        i2 = torch.zeros(1, 8, dtype=torch.long)
        export_one(emb, args.output, "glm_text_embed", i1, i2)

    # ---- Decoder (dynamic seq len) ----
    if want("decoder"):
        print("=== Decoder ===")
        dec = load(Decoder(), args.input, "decoder")
        L1, L2 = 16, 8
        i1 = (torch.randn(1, L1, 1536), build_causal_mask(L1), torch.randn(1, L1, 64), torch.randn(1, L1, 64))
        i2 = (torch.randn(1, L2, 1536), build_causal_mask(L2), torch.randn(1, L2, 64), torch.randn(1, L2, 64))
        export_one(dec, args.output, "glm_decoder", i1, i2)

    # ---- LM Head (fixed single token) ----
    if want("lm_head"):
        print("=== LM Head ===")
        head = load(LMHead(), args.input, "lm_head")
        i1 = torch.randn(1, 1, 1536)
        export_one(head, args.output, "glm_lm_head", i1)

    print(f"\nncnn models -> {args.output}/")


if __name__ == "__main__":
    main()
