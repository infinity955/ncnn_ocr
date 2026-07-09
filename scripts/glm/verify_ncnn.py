#!/usr/bin/env python3
"""
Verify the CONVERTED ncnn models against the PyTorch clean modules on identical
inputs. This is the ncnn-level parity gate (catches pnnx/ncnn op-mapping errors
that PyTorch parity cannot, e.g. batch-axis broadcast, RoPE layout).

Runs each ncnn model via the ncnn python API (blob layout mirrors the C++ contract
and the pnnx-generated *_ncnn.py) and compares to the eager module output.

Usage:
    .venv-glm/Scripts/python.exe scripts/glm/verify_ncnn.py
"""
import os
import numpy as np
import torch
import ncnn

from modules import VisionEncoder, TextEmbed, Decoder, LMHead, build_causal_mask

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "ts")
NC = os.path.join(HERE, "ncnn")


def load(mod, name):
    mod.load_state_dict(torch.load(os.path.join(TS, name + ".pt")))
    return mod.eval()


def run_ncnn(name, blobs, out_name="out0"):
    """blobs: list of (blob_name, numpy_array already in ncnn layout)."""
    with ncnn.Net() as net:
        net.load_param(os.path.join(NC, name + ".ncnn.param").replace("\\", "/"))
        net.load_model(os.path.join(NC, name + ".ncnn.bin").replace("\\", "/"))
        with net.create_extractor() as ex:
            for bn, arr in blobs:
                ex.input(bn, ncnn.Mat(arr).clone())
            _, out = ex.extract(out_name)
            return np.array(out)


def d(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.abs(a.reshape(-1) - b.reshape(-1)).max())


@torch.no_grad()
def main():
    torch.manual_seed(0)
    res = {}

    # ---- text_embed ----
    emb = load(TextEmbed(), "text_embed")
    ids = torch.randint(0, 59392, (1, 12))
    pt = emb(ids)[0].numpy()                                   # (12, 1536)
    nc = run_ncnn("glm_text_embed", [("in0", ids[0].numpy().astype(np.int32))])
    res["text_embed"] = d(pt, nc)

    # ---- lm_head ----
    head = load(LMHead(), "lm_head")
    h = torch.randn(1, 1, 1536)
    pt = head(h)[0].numpy()                                    # (1, 59392)
    nc = run_ncnn("glm_lm_head", [("in0", h[0].numpy())])
    res["lm_head"] = d(pt, nc)

    # ---- decoder ----
    dec = load(Decoder(), "decoder")
    L = 12
    embeds = torch.randn(1, L, 1536)
    mask = build_causal_mask(L)                                # (1,1,L,L)
    cos = torch.randn(1, L, 64); sin = torch.randn(1, L, 64)
    pt = dec(embeds, mask, cos, sin)[0].numpy()                # (L, 1536)
    nc = run_ncnn("glm_decoder", [("in0", embeds[0].numpy()),
                                  ("in1", mask[0].numpy()),
                                  ("in2", cos[0].numpy()),
                                  ("in3", sin[0].numpy())])
    res["decoder"] = d(pt, nc)

    # ---- vision ----
    vis = load(VisionEncoder(), "vision_encoder")
    from modules import build_vision_rope_cos_sin
    nph, npw = 4, 4
    N = nph * npw
    strip = torch.randn(1, 3, 14, 14 * N)
    vcos, vsin = build_vision_rope_cos_sin(nph, npw, 2, rope_dim=32)
    pt = vis(strip, vcos, vsin).numpy()                        # (N/4, 1536)
    nc = run_ncnn("glm_vision", [("in0", strip[0].numpy()),
                                 ("in1", vcos[0].numpy()),
                                 ("in2", vsin[0].numpy())])
    res["vision"] = d(pt, nc)

    print("\n=== ncnn vs PyTorch (maxabs) ===")
    ok = True
    for k, v in res.items():
        s = "OK" if v < 1e-2 else "FAIL"
        ok = ok and v < 1e-2
        print(f"  {k:12s} {v:.3e}  {s}")
    print("\nALL PASS" if ok else "\nSOME FAILED")


if __name__ == "__main__":
    main()
