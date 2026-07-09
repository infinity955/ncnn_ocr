#!/usr/bin/env python3
"""
Verify the clean GLM-OCR modules (modules.py + weights from export_modules.py)
against the real HF submodules. PyTorch-level parity gate before pnnx conversion.

Each module is driven with controlled inputs through the SAME real HF submodule
so the comparison isolates math correctness:
  - VisionEncoder : block-major single-frame strip vs HF core.visual(...).pooler_output
                    (validates Conv3d->Conv2d fold, RoPE, blocks, downsample, merger)
  - TextEmbed     : vs HF embed_tokens
  - Decoder       : random embeds vs HF core.language_model(...).last_hidden_state
  - LMHead        : vs HF lm_head

Usage:
    .venv-glm/Scripts/python.exe scripts/glm/verify_modules.py --model ../glm_ocr_model
"""

import argparse
import os
import sys

import torch

from modules import (
    VisionEncoder, TextEmbed, Decoder, LMHead,
    build_causal_mask, build_text_mrope_cos_sin, build_vision_rope_cos_sin,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "ts")


def diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def load(mod, name):
    mod.load_state_dict(torch.load(os.path.join(TS, name + ".pt")))
    return mod.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    from transformers import AutoModelForImageTextToText
    print(f"Loading HF model from {args.model} ...")
    hf = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
    core = hf.model
    vcfg, tcfg = hf.config.vision_config, hf.config.text_config
    results = {}

    # ---------------- VisionEncoder ----------------
    print("\n[VisionEncoder]")
    vhead = vcfg.hidden_size // vcfg.num_heads
    vis = load(VisionEncoder(hidden=vcfg.hidden_size, depth=vcfg.depth, num_heads=vcfg.num_heads,
                             head_dim=vhead, inter=vcfg.intermediate_size,
                             patch_size=vcfg.patch_size, out_hidden=vcfg.out_hidden_size,
                             spatial_merge=vcfg.spatial_merge_size,
                             merger_ctx=vcfg.out_hidden_size * vcfg.in_channels), "vision_encoder")
    P = vcfg.patch_size
    T = vcfg.temporal_patch_size
    nph, npw = 4, 4                      # 4x4 patches (block-major), N=16 -> 4 merged tokens
    N = nph * npw
    patches = [torch.randn(3, P, P) for _ in range(N)]                    # single-frame, block-major
    # HF input: tile each patch temporally (T identical frames) and flatten -> (N, 3*T*P*P)
    hf_pixels = torch.stack([p.unsqueeze(1).repeat(1, T, 1, 1).reshape(-1) for p in patches], dim=0)
    grid = torch.tensor([[1, nph, npw]], dtype=torch.long)
    hf_vis = core.visual(hf_pixels, grid_thw=grid).pooler_output          # (N/4, 1536)
    strip = torch.cat(patches, dim=-1).unsqueeze(0)                       # (1,3,P,P*N) block-major
    vcos, vsin = build_vision_rope_cos_sin(nph, npw, vcfg.spatial_merge_size, rope_dim=vhead // 2)
    my_vis = vis(strip, vcos, vsin)                                      # (N/4, 1536)  2D
    results["vision"] = diff(my_vis, hf_vis)
    print(f"  shapes my={list(my_vis.shape)} hf={list(hf_vis.shape)}  maxabs={results['vision']:.2e}")

    # ---------------- TextEmbed ----------------
    print("[TextEmbed]")
    emb = load(TextEmbed(tcfg.vocab_size, tcfg.hidden_size), "text_embed")
    ids = torch.randint(0, tcfg.vocab_size, (1, 12))
    results["embed"] = diff(emb(ids), core.language_model.embed_tokens(ids))
    print(f"  maxabs={results['embed']:.2e}")

    # ---------------- Decoder ----------------
    print("[Decoder]")
    thead = getattr(tcfg, "head_dim", tcfg.hidden_size // tcfg.num_attention_heads)
    dec = load(Decoder(num_layers=tcfg.num_hidden_layers, hidden=tcfg.hidden_size,
                       n_q=tcfg.num_attention_heads, n_kv=tcfg.num_key_value_heads,
                       head_dim=thead, inter=tcfg.intermediate_size), "decoder")
    L = 10
    embeds = torch.randn(1, L, tcfg.hidden_size)
    pos = torch.arange(L).unsqueeze(0)                                    # (1,L) -> HF expands to (3,1,L)
    hf_dec = core.language_model(inputs_embeds=embeds, position_ids=pos,
                                 use_cache=False).last_hidden_state       # (1,L,1536)
    pos3 = torch.arange(L).unsqueeze(0).repeat(3, 1)                      # (3,L) text-only
    sec = tcfg.rope_parameters["mrope_section"] if getattr(tcfg, "rope_parameters", None) else [16, 24, 24]
    theta = tcfg.rope_parameters.get("rope_theta", 10000.0) if getattr(tcfg, "rope_parameters", None) else 10000.0
    tcos, tsin = build_text_mrope_cos_sin(pos3, head_dim=thead, mrope_section=sec, rope_theta=theta)
    my_dec = dec(embeds, build_causal_mask(L), tcos, tsin)
    results["decoder"] = diff(my_dec, hf_dec)
    print(f"  maxabs={results['decoder']:.2e}")

    # ---------------- LMHead ----------------
    print("[LMHead]")
    head = load(LMHead(tcfg.hidden_size, tcfg.vocab_size), "lm_head")
    h = torch.randn(1, 1, tcfg.hidden_size)
    results["lmhead"] = diff(head(h), hf.lm_head(h))
    print(f"  maxabs={results['lmhead']:.2e}")

    # ---------------- Summary ----------------
    print("\n=== Summary (tol={:.0e}) ===".format(args.tol))
    ok = True
    for k, v in results.items():
        status = "OK" if v < args.tol else "FAIL"
        ok = ok and (v < args.tol)
        print(f"  {k:10s} maxabs={v:.2e}  {status}")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
