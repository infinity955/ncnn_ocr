#!/usr/bin/env python3
"""
Convert PaddleOCR-VL clean modules to ncnn via pnnx (adapted from scripts/glm/).

Vision has 4 inputs: strip (1,3,14,14*N) + pos_embed (1,N,1152) + cos/sin (1,N,36).
Output names match paddleocr_vl_model.json: pdvl_{vision,text_embed,decoder,lm_head}.ncnn.*

Usage:
  .venv-paddle/Scripts/python.exe scripts/paddleocr_vl/export_vision_ncnn.py [--only vision,...]
"""
import argparse
import os
import torch
import pnnx

from modules import VisionModel, TextEmbed, Decoder, LMHead, build_causal_mask

HERE = os.path.dirname(os.path.abspath(__file__))


def load(mod, ts, name):
    mod.load_state_dict(torch.load(os.path.join(ts, name + ".pt"))); return mod.eval()


def export_one(mod, out_dir, out_name, inputs, inputs2=None):
    j = lambda *p: os.path.join(*p).replace("\\", "/")   # fwd slashes (pnnx path escaping)
    pt = j(out_dir, out_name + ".pt")
    torch.jit.trace(mod, inputs, check_trace=False).save(pt)
    kw = dict(inputs=inputs, optlevel=2, fp16=False, device="cpu",
              pnnxparam=j(out_dir, out_name + ".pnnx.param"), pnnxbin=j(out_dir, out_name + ".pnnx.bin"),
              pnnxpy=j(out_dir, out_name + "_pnnx.py"), pnnxonnx=j(out_dir, out_name + ".pnnx.onnx"),
              ncnnparam=j(out_dir, out_name + ".ncnn.param"), ncnnbin=j(out_dir, out_name + ".ncnn.bin"),
              ncnnpy=j(out_dir, out_name + "_ncnn.py"))
    if inputs2 is not None:
        kw["inputs2"] = inputs2
    try:
        pnnx.export(mod, pt, **kw)
    except Exception as e:
        print(f"  (pnnx post-step: {type(e).__name__})")
    ok = os.path.exists(kw["ncnnparam"]) and os.path.exists(kw["ncnnbin"])
    print(f"  {out_name}: {'OK' if ok else 'MISSING'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default=os.path.join(HERE, "ts"))
    ap.add_argument("--output", "-o", default=os.path.join(HERE, "ncnn"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    want = lambda n: (not only) or (n in only)

    if want("vision"):
        print("=== Vision ===")
        vis = load(VisionModel(), args.input, "vision")
        N1, N2 = 64, 16
        i1 = (torch.randn(1, 3, 14, 14 * N1), torch.randn(1, N1, 1152), torch.randn(1, N1, 36), torch.randn(1, N1, 36))
        i2 = (torch.randn(1, 3, 14, 14 * N2), torch.randn(1, N2, 1152), torch.randn(1, N2, 36), torch.randn(1, N2, 36))
        export_one(vis, args.output, "pdvl_vision", i1, i2)

    if want("text_embed"):
        print("=== Text Embed ===")
        emb = load(TextEmbed(), args.input, "text_embed")
        export_one(emb, args.output, "pdvl_text_embed", torch.zeros(1, 16, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long))

    if want("decoder"):
        print("=== Decoder ===")
        dec = load(Decoder(), args.input, "decoder")
        L1, L2 = 16, 8
        i1 = (torch.randn(1, L1, 1024), build_causal_mask(L1), torch.randn(1, L1, 64), torch.randn(1, L1, 64))
        i2 = (torch.randn(1, L2, 1024), build_causal_mask(L2), torch.randn(1, L2, 64), torch.randn(1, L2, 64))
        export_one(dec, args.output, "pdvl_decoder", i1, i2)

    if want("lm_head"):
        print("=== LM Head ===")
        head = load(LMHead(), args.input, "lm_head")
        export_one(head, args.output, "pdvl_lm_head", torch.randn(1, 1, 1024))

    print(f"\nncnn -> {args.output}/")


if __name__ == "__main__":
    main()
