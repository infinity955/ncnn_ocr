#!/usr/bin/env python3
"""
Export PaddleOCR-VL-1.6 HF weights into the clean modules (modules.py) + save state_dicts.

Loads the real model (transformers 4.55, trust_remote_code), copies weights into
VisionModel / TextEmbed / Decoder / LMHead with the real module names, and dumps the
learned vision position_embedding base (27x27x1152) as pos_embed.bin for C++ interpolation.

Usage:
  .venv-paddle/Scripts/python.exe scripts/paddleocr_vl/export_modules.py --model <snapshot_dir>
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from modules import VisionModel, TextEmbed, Decoder, LMHead

HERE = os.path.dirname(os.path.abspath(__file__))


def cp(dst, src):
    dst.weight.data.copy_(src.weight.data)
    if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
        dst.bias.data.copy_(src.bias.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--output", "-o", default=os.path.join(HERE, "ts"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from transformers import AutoModelForCausalLM
    print(f"Loading {args.model} ...")
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu").eval()
    vt = hf.visual.vision_model        # PaddleOCRVisionTransformer
    dec_src = hf.model                 # Ernie4_5Model
    proj = hf.mlp_AR                   # Projector

    # ---------------- VisionModel (encoder + projector) ----------------
    print("[VisionModel]")
    vis = VisionModel().eval()
    cp(vis.patch_embedding, vt.embeddings.patch_embedding)
    for i, blk in enumerate(vis.blocks):
        s = vt.encoder.layers[i]
        cp(blk.layer_norm1, s.layer_norm1)
        cp(blk.layer_norm2, s.layer_norm2)
        cp(blk.self_attn.q_proj, s.self_attn.q_proj)
        cp(blk.self_attn.k_proj, s.self_attn.k_proj)
        cp(blk.self_attn.v_proj, s.self_attn.v_proj)
        cp(blk.self_attn.out_proj, s.self_attn.out_proj)
        cp(blk.mlp.fc1, s.mlp.fc1)
        cp(blk.mlp.fc2, s.mlp.fc2)
    cp(vis.post_layernorm, vt.post_layernorm)
    cp(vis.pre_norm, proj.pre_norm)
    cp(vis.linear_1, proj.linear_1)
    cp(vis.linear_2, proj.linear_2)
    torch.save(vis.state_dict(), os.path.join(args.output, "vision.pt"))
    print("  saved vision.pt")

    # learned position embedding base (729=27x27, 1152) -> pos_embed.bin for C++ interpolation
    pe = vt.embeddings.position_embedding.weight.data.float().numpy()  # (729,1152)
    pe.astype(np.float32).tofile(os.path.join(args.output, "pos_embed.bin"))
    print(f"  saved pos_embed.bin {pe.shape}")

    # ---------------- TextEmbed ----------------
    print("[TextEmbed]")
    emb = TextEmbed().eval()
    emb.embed.weight.data.copy_(dec_src.embed_tokens.weight.data)
    torch.save(emb.state_dict(), os.path.join(args.output, "text_embed.pt"))

    # ---------------- Decoder ----------------
    print("[Decoder]")
    dec = Decoder().eval()
    for i, lyr in enumerate(dec.layers):
        s = dec_src.layers[i]
        lyr.input_layernorm.weight.data.copy_(s.input_layernorm.weight.data)
        lyr.post_attention_layernorm.weight.data.copy_(s.post_attention_layernorm.weight.data)
        cp(lyr.self_attn.q_proj, s.self_attn.q_proj)
        cp(lyr.self_attn.k_proj, s.self_attn.k_proj)
        cp(lyr.self_attn.v_proj, s.self_attn.v_proj)
        cp(lyr.self_attn.o_proj, s.self_attn.o_proj)
        cp(lyr.mlp.gate_proj, s.mlp.gate_proj)
        cp(lyr.mlp.up_proj, s.mlp.up_proj)
        cp(lyr.mlp.down_proj, s.mlp.down_proj)
    dec.norm.weight.data.copy_(dec_src.norm.weight.data)
    torch.save(dec.state_dict(), os.path.join(args.output, "decoder.pt"))

    # ---------------- LMHead ----------------
    print("[LMHead]")
    head = LMHead().eval()
    head.linear.weight.data.copy_(hf.lm_head.weight.data)
    torch.save(head.state_dict(), os.path.join(args.output, "lm_head.pt"))

    print(f"\nAll saved to {args.output}/")


if __name__ == "__main__":
    main()
