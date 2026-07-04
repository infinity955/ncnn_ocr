"""Re-export decoder / vision / lm_head as TRACED TorchScript (pnnx-friendly), into ts_pnnx/.
Weights are pulled from the scripted ts/*.pt state_dicts (no HF reload).

pnnx crashes at pass_level1 on our *scripted* multi-op modules, but converts the *traced*
graphs cleanly, so the ncnn/ artifacts are built from these traced copies. The primary
TorchScript deliverable stays the scripted dynamic ts/*.pt (produced by export_modules.py).

Note: tracing bakes the vision position-embedding interpolate to the traced H,W, so the
traced vision (and thus its ncnn) is fixed-resolution (384x896 here).
"""
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
import os
import torch
import hyocr_modules_modify as M

TS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts_pnnx")
os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def retrace(name, build, example):
    sm = torch.jit.load(os.path.join(TS, name + ".pt")).eval()
    mod = build().eval()
    mod.load_state_dict(sm.state_dict())
    tr = torch.jit.trace(mod, example, check_trace=False)
    out = tr(*example) if isinstance(example, tuple) else tr(example)
    tr.save(os.path.join(OUT, name + ".pt"))
    shp = tuple(out.shape) if hasattr(out, "shape") else None
    print(f"[retrace] {name} -> {shp}")


def main():
    # decoder: (hidden, mask, cos, sin) with L=8; reshapes use size() so trace stays dynamic
    L = 8
    cos, sin = M.build_cos_sin(torch.arange(L).reshape(1, 1, L).repeat(1, 4, 1))
    dec_ex = (torch.randn(1, L, 1024), M.build_causal_mask(L), cos, sin)
    retrace("decoder", M.Decoder, dec_ex)

    # text embed: (1,L) int64 ids
    retrace("text_embed", M.TextEmbed, torch.zeros(1, 8, dtype=torch.long))

    # lm_head: (1,L,1024)
    retrace("lm_head", M.LMHead, torch.randn(1, 8, 1024))

    # vision: (1,3,H,W) -- interpolate scale bakes to this H,W under trace (fixed-res in ncnn)
    retrace("vision_encoder", M.VisionEncoder, torch.randn(1, 3, 384, 896))

    print("[done]")


if __name__ == "__main__":
    main()
