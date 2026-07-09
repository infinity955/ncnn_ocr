#!/usr/bin/env python3
"""ncnn-vs-PyTorch parity for PaddleOCR-VL converted models (run with .venv-glm python: has ncnn)."""
import os
import numpy as np
import torch
import ncnn

from modules import (VisionModel, TextEmbed, Decoder, LMHead, build_causal_mask,
                     build_text_mrope_cos_sin, build_vision_rope_cos_sin)

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "ts")
NC = os.path.join(HERE, "ncnn")


def load(m, name):
    m.load_state_dict(torch.load(os.path.join(TS, name + ".pt"))); return m.eval()


def run_ncnn(name, blobs, out="out0"):
    with ncnn.Net() as net:
        net.load_param(os.path.join(NC, name + ".ncnn.param").replace("\\", "/"))
        net.load_model(os.path.join(NC, name + ".ncnn.bin").replace("\\", "/"))
        with net.create_extractor() as ex:
            for bn, arr in blobs:
                ex.input(bn, ncnn.Mat(arr).clone())
            _, o = ex.extract(out)
            return np.array(o)


def d(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a.reshape(-1) - b.reshape(-1)).max())


@torch.no_grad()
def main():
    torch.manual_seed(0)
    res = {}

    emb = load(TextEmbed(), "text_embed")
    ids = torch.randint(0, 103424, (1, 12))
    res["text_embed"] = d(emb(ids)[0].numpy(), run_ncnn("pdvl_text_embed", [("in0", ids[0].numpy().astype(np.int32))]))

    head = load(LMHead(), "lm_head")
    h = torch.randn(1, 1, 1024)
    res["lm_head"] = d(head(h)[0].numpy(), run_ncnn("pdvl_lm_head", [("in0", h[0].numpy())]))

    dec = load(Decoder(), "decoder")
    L = 12
    e = torch.randn(1, L, 1024); mask = build_causal_mask(L)
    cos, sin = build_text_mrope_cos_sin(torch.arange(L).unsqueeze(0).repeat(3, 1), head_dim=128,
                                        mrope_section=[16, 24, 24], rope_theta=500000.0)  # real bounded cos/sin
    res["decoder"] = d(dec(e, mask, cos, sin)[0].numpy(),
                       run_ncnn("pdvl_decoder", [("in0", e[0].numpy()), ("in1", mask[0].numpy()),
                                                 ("in2", cos[0].numpy()), ("in3", sin[0].numpy())]))

    vis = load(VisionModel(), "vision")
    gh, gw = 4, 6; N = gh * gw
    strip = torch.randn(1, 3, 14, 14 * N); pe = torch.randn(1, N, 1152)
    vc, vs = build_vision_rope_cos_sin(gh, gw, head_dim=72, block_major=True)   # real bounded 2D rope
    res["vision"] = d(vis(strip, pe, vc, vs).numpy(),
                      run_ncnn("pdvl_vision", [("in0", strip[0].numpy()), ("in1", pe[0].numpy()),
                                               ("in2", vc[0].numpy()), ("in3", vs[0].numpy())]))

    print("\n=== ncnn vs PyTorch (maxabs) ===")
    ok = True
    for k, v in res.items():
        s = "OK" if v < 1e-2 else "FAIL"; ok = ok and v < 1e-2
        print(f"  {k:12s} {v:.3e}  {s}")
    print("\nALL PASS" if ok else "\nSOME FAILED")


if __name__ == "__main__":
    main()
