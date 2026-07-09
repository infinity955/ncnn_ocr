#!/usr/bin/env python3
"""
Re-export GLM-OCR modules as traced TorchScript for pnnx.

pnnx requires traced (not scripted) graphs. This script loads the
scripted .pt files from ts/, rebuilds eager modules, and re-exports
via torch.jit.trace().

Usage:
    python trace_export.py
    python trace_export.py --input ts --output ts_pnnx
"""

import argparse
import os
import sys

import torch

from modules import (
    VisionEncoder, TextEmbed, Decoder, LMHead,
    build_causal_mask,
)

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/glm


def main():
    parser = argparse.ArgumentParser(description="Re-export GLM-OCR as traced TorchScript")
    parser.add_argument("--input", "-i", type=str, default=os.path.join(HERE, "ts"),
                        help="Input directory with scripted .pt files")
    parser.add_argument("--output", "-o", type=str, default=os.path.join(HERE, "ts_pnnx"),
                        help="Output directory for traced .pt files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ============================================================
    # VisionEncoder
    # ============================================================
    vis_path = os.path.join(args.input, "vision_encoder.pt")
    if os.path.exists(vis_path):
        print("--- Tracing VisionEncoder ---")
        # Build from state dict
        scripted = torch.jit.load(vis_path, map_location="cpu")
        state = scripted.state_dict()
        vis = VisionEncoder(
            hidden=state["patch_embed.weight"].shape[0],
            depth=len([k for k in state if k.startswith("blocks.") and k.endswith(".norm1.weight")]),
            num_heads=state["blocks.0.attn.qkv.weight"].shape[0] // (3 * state["blocks.0.attn.o_proj.weight"].shape[1]),
            inter=state["blocks.0.mlp.gate_proj.weight"].shape[0],
            patch_size=14,
            out_hidden=state["proj.weight"].shape[0],
        )
        vis.load_state_dict(state, strict=False)
        vis.eval()

        # Trace with fixed image size
        N = 16  # number of patches for trace
        dummy_strip = torch.randn(1, 3, 14, 14 * N)
        dummy_cos = torch.randn(1, N, 64)
        dummy_sin = torch.randn(1, N, 64)
        traced = torch.jit.trace(vis, (dummy_strip, dummy_cos, dummy_sin), check_trace=False)
        traced.save(os.path.join(args.output, "vision_encoder.pt"))
        print("  Saved vision_encoder.pt")

    # ============================================================
    # TextEmbed
    # ============================================================
    emb_path = os.path.join(args.input, "text_embed.pt")
    if os.path.exists(emb_path):
        print("--- Tracing TextEmbed ---")
        scripted = torch.jit.load(emb_path, map_location="cpu")
        state = scripted.state_dict()
        emb = TextEmbed(
            vocab_size=state["embed.weight"].shape[0],
            hidden_size=state["embed.weight"].shape[1],
        )
        emb.load_state_dict(state)
        emb.eval()

        dummy_ids = torch.zeros(1, 8, dtype=torch.long)
        traced = torch.jit.trace(emb, dummy_ids, check_trace=False)
        traced.save(os.path.join(args.output, "text_embed.pt"))
        print("  Saved text_embed.pt")

    # ============================================================
    # Decoder
    # ============================================================
    dec_path = os.path.join(args.input, "decoder.pt")
    if os.path.exists(dec_path):
        print("--- Tracing Decoder ---")
        scripted = torch.jit.load(dec_path, map_location="cpu")
        state = scripted.state_dict()
        num_layers = len([k for k in state if k.startswith("layers.") and k.endswith(".norm1.weight")])
        layer0 = "layers.0."
        hidden = state[layer0 + "norm1.weight"].shape[0]
        n_q = state[layer0 + "attn.q_proj.weight"].shape[0] // 128
        n_kv = state[layer0 + "attn.k_proj.weight"].shape[0] // 128
        inter = state[layer0 + "mlp.gate_proj.weight"].shape[0]

        dec = Decoder(num_layers=num_layers, hidden=hidden, n_q=n_q, n_kv=n_kv, inter=inter)
        dec.load_state_dict(state)
        dec.eval()

        L = 8
        dummy_embeds = torch.randn(1, L, hidden)
        dummy_mask = build_causal_mask(L)
        dummy_cos = torch.randn(1, L, 64)
        dummy_sin = torch.randn(1, L, 64)
        traced = torch.jit.trace(dec, (dummy_embeds, dummy_mask, dummy_cos, dummy_sin), check_trace=False)
        traced.save(os.path.join(args.output, "decoder.pt"))
        print("  Saved decoder.pt")

    # ============================================================
    # LMHead
    # ============================================================
    head_path = os.path.join(args.input, "lm_head.pt")
    if os.path.exists(head_path):
        print("--- Tracing LMHead ---")
        scripted = torch.jit.load(head_path, map_location="cpu")
        state = scripted.state_dict()
        head = LMHead(
            hidden=state["linear.weight"].shape[1],
            vocab=state["linear.weight"].shape[0],
        )
        head.load_state_dict(state)
        head.eval()

        dummy_hidden = torch.randn(1, 1, head.linear.weight.shape[1])
        traced = torch.jit.trace(head, dummy_hidden, check_trace=False)
        traced.save(os.path.join(args.output, "lm_head.pt"))
        print("  Saved lm_head.pt")

    print(f"\nAll traced modules saved to: {args.output}/")


if __name__ == "__main__":
    main()