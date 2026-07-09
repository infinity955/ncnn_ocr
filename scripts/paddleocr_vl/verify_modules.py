#!/usr/bin/env python3
"""
Verify clean PaddleOCR-VL modules (modules.py + weights) vs the real HF submodules.
PyTorch parity gate before pnnx.

  TextEmbed / Decoder / LMHead : controlled inputs vs hf.model / hf.lm_head
  VisionModel                  : synthetic patches vs hf.visual(...) + hf.mlp_AR(...)

Usage:
  .venv-paddle/Scripts/python.exe scripts/paddleocr_vl/verify_modules.py --model <snapshot_dir>
"""
import argparse
import os
import sys
import numpy as np
import torch

from modules import (VisionModel, TextEmbed, Decoder, LMHead,
                     build_causal_mask, build_text_mrope_cos_sin, build_vision_rope_cos_sin)

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "ts")


def load(m, name):
    m.load_state_dict(torch.load(os.path.join(TS, name + ".pt"))); return m.eval()


def d(a, b):
    return (a.float() - b.float()).abs().max().item()


def blockmajor_perm(gh, gw, merge=2):
    """block-major seq position -> raster index (h*gw+w)."""
    perm = []
    for bh in range(gh // merge):
        for bw in range(gw // merge):
            for mh in range(merge):
                for mw in range(merge):
                    h, w = bh * merge + mh, bw * merge + mw
                    perm.append(h * gw + w)
    return perm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM
    print(f"Loading {args.model} ...")
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu").eval()
    res = {}

    # ---- TextEmbed ----
    emb = load(TextEmbed(), "text_embed")
    ids = torch.randint(0, 103424, (1, 12))
    res["embed"] = d(emb(ids), hf.model.embed_tokens(ids))

    # ---- Decoder ----
    # Drive HF's real layers/rotary_emb/norm manually (the model.forward path hits a
    # create_causal_mask kwarg mismatch in transformers 4.55.0). Uses HF's actual mrope.
    dec = load(Decoder(), "decoder")
    L = 10
    embeds = torch.randn(1, L, 1024)
    pos3 = torch.arange(L).view(1, 1, L).expand(3, 1, L).contiguous()
    cmask = build_causal_mask(L)
    hs = embeds
    pos_emb = hf.model.rotary_emb(hs, pos3)
    for layer in hf.model.layers:
        out = layer(hs, attention_mask=cmask, position_ids=pos3, position_embeddings=pos_emb)
        hs = out[0] if isinstance(out, (tuple, list)) else out
    hf_out = hf.model.norm(hs)
    tcos, tsin = build_text_mrope_cos_sin(torch.arange(L).unsqueeze(0).repeat(3, 1), head_dim=128,
                                          mrope_section=[16, 24, 24], rope_theta=500000.0)
    my_out = dec(embeds, cmask, tcos, tsin)
    res["decoder"] = d(my_out, hf_out)

    # ---- LMHead ----
    head = load(LMHead(), "lm_head")
    h = torch.randn(1, 1, 1024)
    res["lmhead"] = d(head(h), hf.lm_head(h))

    # ---- VisionModel ----
    vis = load(VisionModel(), "vision")
    gh, gw = 4, 6           # grid (even), N=24 patches -> 6 merged tokens
    N = gh * gw
    patches = torch.randn(N, 3, 14, 14)                        # raster patches
    # HF reference
    pv = patches.unsqueeze(0)                                  # (1,N,3,14,14)
    grid = [(1, gh, gw)]
    sig_pos = torch.arange(N) % (gh * gw)
    cu = torch.tensor([0, N], dtype=torch.int32)
    samp = torch.zeros(N, dtype=torch.int64)
    vout = hf.visual(pixel_values=pv, image_grid_thw=grid, position_ids=sig_pos,
                     vision_return_embed_list=True, interpolate_pos_encoding=True,
                     sample_indices=samp, cu_seqlens=cu, return_pooler_output=False,
                     use_rope=True, window_size=-1)
    ref = torch.cat(hf.mlp_AR(vout.last_hidden_state, grid), dim=0)   # (N/4, 1024) raster-merged
    # My inputs (block-major)
    perm = blockmajor_perm(gh, gw)
    strip = torch.cat([patches[j] for j in perm], dim=-1).unsqueeze(0)    # (1,3,14,14*N) block-major
    # pos_embed: bilinear-interp 27x27 base -> (gh,gw) raster, then reorder block-major
    pe_base = hf.visual.vision_model.embeddings.position_embedding.weight  # (729,1152)
    import torch.nn.functional as F
    pe = pe_base.reshape(1, 27, 27, 1152).permute(0, 3, 1, 2)
    pe = F.interpolate(pe, size=(gh, gw), mode="bilinear", align_corners=False)
    pe = pe.permute(0, 2, 3, 1).reshape(gh * gw, 1152)                    # raster
    pe_bm = pe[perm].unsqueeze(0)                                          # block-major (1,N,1152)
    vc, vs = build_vision_rope_cos_sin(gh, gw, head_dim=72, block_major=True)   # (1,N,36)
    my_vis = vis(strip, pe_bm, vc, vs)                                     # (N/4,1024) block-major-merged (=raster-merged order)
    res["vision"] = d(my_vis, ref)

    print("\n=== Summary (tol={:.0e}) ===".format(args.tol))
    ok = True
    for k, v in res.items():
        s = "OK" if v < args.tol else "FAIL"; ok = ok and v < args.tol
        print(f"  {k:8s} {v:.2e}  {s}")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
