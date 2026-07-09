"""Load HunyuanOCR HF weights into the 4 clean modules, then export TorchScript (.pt), fp32."""
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
import os
import torch
import torch.nn.functional as F
from transformers import HunYuanVLForConditionalGeneration

import modules_modify as M

# MODEL_PATH env var overrides; default: ../../../hunyuanocrmodel relative to this script (pnnx root)
_MODEL_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "hunyuanocrmodel"))
MODEL = os.environ.get("MODEL_PATH", _MODEL_DEFAULT)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts")
os.makedirs(OUT, exist_ok=True)


def cp(dst, src, bias=True):
    dst.weight.data.copy_(src.weight.data)
    if bias:
        dst.bias.data.copy_(src.bias.data)


@torch.no_grad()
def main():
    print("[info] loading HF model (fp32, cpu) ...")
    hf = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, local_files_only=True, attn_implementation="eager"
    ).eval()
    vit, llm, lm_head = hf.vit, hf.model, hf.lm_head

    # ---------- module 1: vision ----------
    vis = M.VisionEncoder().eval()
    cp(vis.patch_embedding, vit.embeddings.patch_embedding, bias=True)
    pe = vit.embeddings.position_embedding.weight[1:, :]  # (16384,1152)
    edge = 128  # sqrt(16384)
    pos_base = pe.reshape(edge, edge, vis.hidden).permute(2, 0, 1).contiguous()  # (1152,128,128)
    pos_file = os.path.join(OUT, "pos_embed.bin")
    pos_base.numpy().tofile(pos_file)
    print(f"  saved pos_embed {tuple(pos_base.shape)} -> {pos_file}")
    for i, (d, s) in enumerate(zip(vis.layers, vit.layers)):
        cp(d.self_attn.q_proj, s.self_attn.q_proj)
        cp(d.self_attn.k_proj, s.self_attn.k_proj)
        cp(d.self_attn.v_proj, s.self_attn.v_proj)
        cp(d.self_attn.o_proj, s.self_attn.o_proj)
        cp(d.mlp.dense_h_to_4h, s.mlp.dense_h_to_4h)
        cp(d.mlp.dense_4h_to_h, s.mlp.dense_4h_to_h)
        cp(d.input_layernorm, s.input_layernorm, bias=True)
        cp(d.post_attention_layernorm, s.post_attention_layernorm, bias=True)
    cp(vis.perceive.proj[0], vit.perceive.proj[0])   # Conv2d
    cp(vis.perceive.proj[2], vit.perceive.proj[2])   # Conv2d
    cp(vis.perceive.mlp, vit.perceive.mlp)
    vis.perceive.image_newline.data.copy_(vit.perceive.image_newline.data)
    vis.perceive.image_begin.data.copy_(vit.perceive.image_begin.data)
    vis.perceive.image_end.data.copy_(vit.perceive.image_end.data)
    vis.perceive.before_rms.weight.data.copy_(vit.perceive.before_rms.weight.data)
    vis.perceive.after_rms.weight.data.copy_(vit.perceive.after_rms.weight.data)

    # ---------- module 2: text embed ----------
    emb = M.TextEmbed().eval()
    emb.embed_tokens.weight.data.copy_(llm.embed_tokens.weight.data)

    # ---------- module 3: decoder ----------
    dec = M.Decoder().eval()
    for d, s in zip(dec.layers, llm.layers):
        cp(d.self_attn.q_proj, s.self_attn.q_proj, bias=False)
        cp(d.self_attn.k_proj, s.self_attn.k_proj, bias=False)
        cp(d.self_attn.v_proj, s.self_attn.v_proj, bias=False)
        cp(d.self_attn.o_proj, s.self_attn.o_proj, bias=False)
        d.self_attn.query_layernorm.weight.data.copy_(s.self_attn.query_layernorm.weight.data)
        d.self_attn.key_layernorm.weight.data.copy_(s.self_attn.key_layernorm.weight.data)
        cp(d.mlp.gate_proj, s.mlp.gate_proj, bias=False)
        cp(d.mlp.up_proj, s.mlp.up_proj, bias=False)
        cp(d.mlp.down_proj, s.mlp.down_proj, bias=False)
        d.input_layernorm.weight.data.copy_(s.input_layernorm.weight.data)
        d.post_attention_layernorm.weight.data.copy_(s.post_attention_layernorm.weight.data)
    dec.norm.weight.data.copy_(llm.norm.weight.data)

    # ---------- module 4: lm head ----------
    head = M.LMHead().eval()
    head.lm_head.weight.data.copy_(lm_head.weight.data)

    # ---------- quick eager sanity ----------
    print("[info] eager sanity forward ...")
    px = torch.randn(1, 3, 64, 96)
    # pos_embed must match patch grid: gh=H/16, gw=W/16
    ppos = F.interpolate(pos_base.unsqueeze(0), size=[4, 6], mode="bilinear", align_corners=False)
    vout = vis(px, ppos); print("  vision:", tuple(vout.shape))
    ids = torch.randint(0, 120818, (1, 20))
    eout = emb(ids); print("  embed :", tuple(eout.shape))
    L = 20
    cos, sin = M.build_cos_sin(torch.arange(L).reshape(1, 1, L).repeat(1, 4, 1))
    mask = M.build_causal_mask(L)
    dout = dec(eout, mask, cos, sin); print("  decode:", tuple(dout.shape))
    hout = head(dout[:, -1:, :]); print("  lmhead:", tuple(hout.shape))

    # ---------- script + save ----------
    print("[info] scripting + saving ...")
    for name, mod in [("vision_encoder", vis), ("text_embed", emb),
                      ("decoder", dec), ("lm_head", head)]:
        sm = torch.jit.script(mod)
        p = os.path.join(OUT, name + ".pt")
        sm.save(p)
        print(f"  saved {p}")

    print("[done]")


if __name__ == "__main__":
    main()
