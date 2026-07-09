#!/usr/bin/env python3
"""
Export GLM-OCR HF weights into the clean modules (modules.py) and save state_dicts.

Loads the real glm_ocr model (transformers 5.13.0), copies weights into
VisionEncoder / TextEmbed / Decoder / LMHead with the CORRECT real module names,
folding the vision Conv3d patch-embed into a Conv2d (temporal frames are identical
for a single tiled image, so Conv2d weight = sum over the temporal axis).

Outputs ts/{vision_encoder,text_embed,decoder,lm_head}.pt (torch state_dicts).
Downstream trace_export.py / verify_modules.py rebuild eager modules + load_state_dict.

Usage:
    .venv-glm/Scripts/python.exe scripts/glm/export_modules.py --model ../glm_ocr_model
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

from modules import VisionEncoder, TextEmbed, Decoder, LMHead

HERE = os.path.dirname(os.path.abspath(__file__))


def cp_lin(dst: nn.Module, src: nn.Module):
    """Copy Linear/Conv weight (+bias if both present)."""
    dst.weight.data.copy_(src.weight.data)
    if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
        dst.bias.data.copy_(src.bias.data)


def cp_norm(dst: nn.Module, src: nn.Module):
    """Copy norm weight (+bias for LayerNorm)."""
    dst.weight.data.copy_(src.weight.data)
    if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
        dst.bias.data.copy_(src.bias.data)


def perm_interleave_to_half(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder per-head rows so ncnn-friendly contiguous RoPE == HF interleaved RoPE.

    Interleaved pairs (2i, 2i+1) -> contiguous pairs (i, i+head_dim/2).
    w: (n_heads*head_dim, hidden). Applied to q_proj/k_proj weights only.
    """
    hid = w.shape[1]
    W = w.view(n_heads, head_dim, hid)
    idx = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
    return W[:, idx, :].reshape(n_heads * head_dim, hid).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True, help="Path to glm_ocr HF model dir")
    ap.add_argument("--output", "-o", default=os.path.join(HERE, "ts"))
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading HF model from {args.model} ...")
    from transformers import AutoModelForImageTextToText
    hf = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True,
    ).eval()
    core = hf.model
    vsrc = core.visual
    tsrc = core.language_model
    vcfg = hf.config.vision_config
    tcfg = hf.config.text_config
    print(f"Loaded {type(hf).__name__}: vision depth={vcfg.depth}, text layers={tcfg.num_hidden_layers}")

    # ---------------- VisionEncoder ----------------
    print("\n[VisionEncoder]")
    vhead_dim = vcfg.hidden_size // vcfg.num_heads
    vis = VisionEncoder(hidden=vcfg.hidden_size, depth=vcfg.depth, num_heads=vcfg.num_heads,
                        head_dim=vhead_dim, inter=vcfg.intermediate_size, patch_size=vcfg.patch_size,
                        out_hidden=vcfg.out_hidden_size, spatial_merge=vcfg.spatial_merge_size,
                        merger_ctx=vcfg.out_hidden_size * vcfg.in_channels).eval()

    # Conv3d -> Conv2d fold: sum over temporal dim (kernel [T,14,14] -> [14,14])
    w3d = vsrc.patch_embed.proj.weight.data        # (1024, 3, T, 14, 14)
    vis.patch_embed.weight.data.copy_(w3d.sum(dim=2))
    vis.patch_embed.bias.data.copy_(vsrc.patch_embed.proj.bias.data)

    for i, blk in enumerate(vis.blocks):
        s = vsrc.blocks[i]
        cp_norm(blk.norm1, s.norm1)
        cp_norm(blk.norm2, s.norm2)
        cp_lin(blk.attn.qkv, s.attn.qkv)
        cp_lin(blk.attn.proj, s.attn.proj)
        cp_norm(blk.attn.q_norm, s.attn.q_norm)
        cp_norm(blk.attn.k_norm, s.attn.k_norm)
        cp_lin(blk.mlp.gate_proj, s.mlp.gate_proj)
        cp_lin(blk.mlp.up_proj, s.mlp.up_proj)
        cp_lin(blk.mlp.down_proj, s.mlp.down_proj)
    cp_norm(vis.post_layernorm, vsrc.post_layernorm)
    # downsample Conv2d(hidden, out, k=2, s=2) -> Linear(hidden*4, out): reshape weight
    vis.downsample.weight.data.copy_(vsrc.downsample.weight.data.reshape(vcfg.out_hidden_size, -1))
    vis.downsample.bias.data.copy_(vsrc.downsample.bias.data)
    cp_lin(vis.merger.proj, vsrc.merger.proj)
    cp_norm(vis.merger.post_projection_norm, vsrc.merger.post_projection_norm)
    cp_lin(vis.merger.gate_proj, vsrc.merger.gate_proj)
    cp_lin(vis.merger.up_proj, vsrc.merger.up_proj)
    cp_lin(vis.merger.down_proj, vsrc.merger.down_proj)
    torch.save(vis.state_dict(), os.path.join(args.output, "vision_encoder.pt"))
    print(f"  saved vision_encoder.pt (folded Conv3d T={w3d.shape[2]} -> Conv2d)")

    # ---------------- TextEmbed ----------------
    print("[TextEmbed]")
    emb = TextEmbed(tcfg.vocab_size, tcfg.hidden_size).eval()
    emb.embed.weight.data.copy_(tsrc.embed_tokens.weight.data)
    torch.save(emb.state_dict(), os.path.join(args.output, "text_embed.pt"))
    print(f"  saved text_embed.pt (vocab={tcfg.vocab_size}, hidden={tcfg.hidden_size})")

    # ---------------- Decoder ----------------
    print("[Decoder]")
    thead_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // tcfg.num_attention_heads)
    dec = Decoder(num_layers=tcfg.num_hidden_layers, hidden=tcfg.hidden_size,
                  n_q=tcfg.num_attention_heads, n_kv=tcfg.num_key_value_heads,
                  head_dim=thead_dim, inter=tcfg.intermediate_size).eval()
    for i, lyr in enumerate(dec.layers):
        s = tsrc.layers[i]
        cp_norm(lyr.input_layernorm, s.input_layernorm)
        cp_norm(lyr.post_self_attn_layernorm, s.post_self_attn_layernorm)
        cp_norm(lyr.post_attention_layernorm, s.post_attention_layernorm)
        cp_norm(lyr.post_mlp_layernorm, s.post_mlp_layernorm)
        cp_lin(lyr.self_attn.q_proj, s.self_attn.q_proj)
        cp_lin(lyr.self_attn.k_proj, s.self_attn.k_proj)
        # permute q/k rows: interleaved (HF) -> contiguous (ncnn-friendly) RoPE
        lyr.self_attn.q_proj.weight.data.copy_(
            perm_interleave_to_half(s.self_attn.q_proj.weight.data, tcfg.num_attention_heads, thead_dim))
        lyr.self_attn.k_proj.weight.data.copy_(
            perm_interleave_to_half(s.self_attn.k_proj.weight.data, tcfg.num_key_value_heads, thead_dim))
        cp_lin(lyr.self_attn.v_proj, s.self_attn.v_proj)
        cp_lin(lyr.self_attn.o_proj, s.self_attn.o_proj)
        cp_lin(lyr.mlp.gate_up_proj, s.mlp.gate_up_proj)
        cp_lin(lyr.mlp.down_proj, s.mlp.down_proj)
    cp_norm(dec.norm, tsrc.norm)
    torch.save(dec.state_dict(), os.path.join(args.output, "decoder.pt"))
    print(f"  saved decoder.pt (layers={tcfg.num_hidden_layers}, head_dim={thead_dim})")

    # ---------------- LMHead ----------------
    print("[LMHead]")
    head = LMHead(tcfg.hidden_size, tcfg.vocab_size).eval()
    head.linear.weight.data.copy_(hf.lm_head.weight.data)
    torch.save(head.state_dict(), os.path.join(args.output, "lm_head.pt"))
    print(f"  saved lm_head.pt")

    print(f"\nAll state_dicts saved to {args.output}/")


if __name__ == "__main__":
    main()
