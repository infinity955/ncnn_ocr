"""Export the vision encoder to a DYNAMIC-resolution ncnn via pnnx.export with two
different input sizes (DA-V2 technique). The eager model is traced internally by pnnx;
size=[gh,gw] interpolate stays a dynamic ncnn Interp instead of a folded constant.
"""
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
import os
import torch
import pnnx
import hyocr_modules_modify as M

TS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncnn")
os.makedirs(OUTDIR, exist_ok=True)


def main():
    sm = torch.jit.load(os.path.join(TS, "vision_encoder.pt")).eval()
    v = M.VisionEncoder().eval()
    v.load_state_dict(sm.state_dict())

    pt_path = os.path.join(OUTDIR, "vision_encoder.pt")
    # two different H and W (both multiples of 32) -> pnnx marks H,W dynamic.
    # pnnx.export runs a post-conversion self-test by importing the generated _pnnx.py;
    # on Windows that file has an unescaped path and raises SyntaxError AFTER the ncnn
    # files are already written, so ignore it.
    try:
        pnnx.export(
            v, pt_path,
            inputs=torch.rand(1, 3, 384, 896),
            inputs2=torch.rand(1, 3, 224, 448),
            fp16=False,
        )
    except Exception as e:
        print("[warn] pnnx.export post-step raised (ncnn files already written):", repr(e)[:120])
    ncnn_param = pt_path.replace(".pt", ".ncnn.param")
    print("[done] ncnn param exists:", os.path.exists(ncnn_param))


if __name__ == "__main__":
    main()
