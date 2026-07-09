#!/usr/bin/env python3
"""
Clean TorchScript-compatible module definitions for PaddleOCR-VL-1.6.

Mirrors modeling_paddleocr_vl.py (transformers 4.55, trust_remote_code):
  Vision : SigLIP-so400m (27L, hidden 1152, LayerNorm, gelu_tanh, MHA+bias, 2D RoPE)
           + external interpolated pos_embed + PatchMerger/Projector (2x2 -> 1024)
  Text   : ERNIE-4.5 (18L, hidden 1024, GQA 16/2, head_dim 128, separate SwiGLU,
           standard 2-norm pre-norm, contiguous NEOX mRoPE [16,24,24] theta 500000)

ncnn-friendly conventions (learned from GLM):
  - batch=1 literal; q/k/v permute(0,2,1,3) then rope with 3D cos/sin (broadcast over heads)
  - CONTIGUOUS rotate_half everywhere (both vision & text are NEOX-style) -> no step-2 slice
  - RMSNorm/LayerNorm without dtype casts -> 0 aten::to
  - vision cos/sin come in width 36 (module cat->72); text cos/sin width 64 (module cat->128)
  - vision patch fed as block-major strip (1,3,14,14*N); projector merge = consecutive-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ shared
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ============================================================ VISION (SigLIP)
class VisionMLP(nn.Module):
    def __init__(self, hidden=1152, inter=4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter, bias=True)
        self.fc2 = nn.Linear(inter, hidden, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VisionAttention(nn.Module):
    def __init__(self, hidden=1152, num_heads=16, head_dim=72):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden = hidden
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden, bias=True)

    def forward(self, x, cos, sin):
        # x:(1,N,hidden); cos/sin:(1,N,36) -> cat to 72
        N = x.size(1)
        q = self.q_proj(x).reshape(1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        cos2 = torch.cat((cos, cos), dim=-1)  # (1,N,72)
        sin2 = torch.cat((sin, sin), dim=-1)
        q = q * cos2 + rotate_half(q) * sin2
        k = k * cos2 + rotate_half(k) * sin2
        o = F.scaled_dot_product_attention(q, k, v)  # full attention (single image)
        o = o.permute(0, 2, 1, 3).reshape(1, N, self.num_heads * self.head_dim)
        return self.out_proj(o)


class VisionBlock(nn.Module):
    def __init__(self, hidden=1152, num_heads=16, head_dim=72, inter=4304, eps=1e-6):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.self_attn = VisionAttention(hidden, num_heads, head_dim)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = VisionMLP(hidden, inter)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionModel(nn.Module):
    """SigLIP encoder + pre_norm + 2x2 merge + Projector -> (N/4, text_hidden).

    Inputs:
      strip:     (1,3,patch,patch*N)  block-major patches (2x2 blocks consecutive)
      pos_embed: (1,N,1152)           pre-interpolated learned position embedding
      cos,sin:   (1,N,36)             2D vision RoPE (h18|w18)
    Output: (N/4, 1024)
    """
    def __init__(self, hidden=1152, depth=27, num_heads=16, head_dim=72, inter=4304,
                 patch=14, merge=2, out_hidden=1024, eps=1e-6):
        super().__init__()
        self.hidden = hidden
        self.merge = merge * merge
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList([VisionBlock(hidden, num_heads, head_dim, inter, eps) for _ in range(depth)])
        self.post_layernorm = nn.LayerNorm(hidden, eps=eps)
        # Projector
        self.pre_norm = nn.LayerNorm(hidden, eps=1e-5)
        ctx = hidden * self.merge  # 4608
        self.linear_1 = nn.Linear(ctx, ctx, bias=True)
        self.linear_2 = nn.Linear(ctx, out_hidden, bias=True)

    def forward(self, strip, pos_embed, cos, sin):
        x = self.patch_embedding(strip)          # (1,1152,1,N)
        x = x.flatten(2).transpose(1, 2)         # (1,N,1152)
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.post_layernorm(x)
        x = self.pre_norm(x)                     # per-patch LN before merge
        # 2x2 merge (block-major strip -> consecutive-4 grouping)
        x = x.reshape(-1, self.hidden * self.merge)   # (N/4, 4608)
        x = self.linear_2(F.gelu(self.linear_1(x)))   # (N/4, 1024)
        return x


# ============================================================ TEXT (ERNIE-4.5)
class TextEmbed(nn.Module):
    def __init__(self, vocab=103424, hidden=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)

    def forward(self, ids):
        return self.embed(ids)


class DecoderAttention(nn.Module):
    """GQA 16/2, head_dim 128, no q/k norm, contiguous mRoPE (cos/sin width 64 -> 128)."""
    def __init__(self, hidden=1024, n_q=16, n_kv=2, head_dim=128):
        super().__init__()
        self.n_q, self.n_kv, self.head_dim = n_q, n_kv, head_dim
        self.q_per_kv = n_q // n_kv
        self.q_proj = nn.Linear(hidden, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, hidden, bias=False)

    def forward(self, x, mask, cos, sin):
        L = x.size(1)
        q = self.q_proj(x).reshape(1, L, self.n_q, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, L, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, L, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        cos2 = torch.cat((cos, cos), dim=-1)  # (1,L,128)
        sin2 = torch.cat((sin, sin), dim=-1)
        q = q * cos2 + rotate_half(q) * sin2
        k = k * cos2 + rotate_half(k) * sin2
        k = k.repeat_interleave(self.q_per_kv, dim=-3)
        v = v.repeat_interleave(self.q_per_kv, dim=-3)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.permute(0, 2, 1, 3).reshape(1, L, self.n_q * self.head_dim)
        return self.o_proj(o)


class DecoderMLP(nn.Module):
    def __init__(self, hidden=1024, inter=3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Standard 2-norm pre-norm (input_layernorm + post_attention_layernorm)."""
    def __init__(self, hidden=1024, n_q=16, n_kv=2, head_dim=128, inter=3072, eps=1e-5):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = DecoderAttention(hidden, n_q, n_kv, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = DecoderMLP(hidden, inter)

    def forward(self, x, mask, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers=18, hidden=1024, n_q=16, n_kv=2, head_dim=128, inter=3072, eps=1e-5):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(hidden, n_q, n_kv, head_dim, inter, eps) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden, eps)

    def forward(self, embeds, mask, cos, sin):
        x = embeds
        for layer in self.layers:
            x = layer(x, mask, cos, sin)
        return self.norm(x)


class LMHead(nn.Module):
    def __init__(self, hidden=1024, vocab=103424):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        return self.linear(x)


# ============================================================ driver helpers
def build_causal_mask(L):
    return torch.triu(torch.full((1, 1, L, L), float("-inf")), diagonal=1)


def build_text_mrope_cos_sin(position_ids, head_dim=128, mrope_section=None, rope_theta=500000.0):
    """position_ids:(3,L) -> cos,sin:(1,L,head_dim//2). Mirrors apply_multimodal_rotary_pos_emb."""
    if mrope_section is None:
        mrope_section = [16, 24, 24]
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
    freqs = torch.stack([torch.outer(position_ids[a].float(), inv_freq) for a in range(3)], 0)  # (3,L,half)
    parts, off = [], 0
    for i, sec in enumerate(mrope_section):
        parts.append(freqs[i % 3, :, off:off + sec]); off += sec
    freq = torch.cat(parts, dim=-1)
    return freq.cos().unsqueeze(0), freq.sin().unsqueeze(0)


def build_vision_rope_cos_sin(grid_h, grid_w, head_dim=72, rope_theta=10000.0, block_major=True, merge=2):
    """2D vision RoPE. Returns cos,sin:(1,N,head_dim//2=36) = [h18|w18] per patch.

    Patch order: block-major (2x2 blocks consecutive) if block_major else raster.
    """
    rdim = head_dim // 2                    # 36
    half = rdim // 2                        # 18 freqs per axis
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rdim))
    coords = []
    if block_major:
        for gh in range(grid_h // merge):
            for gw in range(grid_w // merge):
                for mh in range(merge):
                    for mw in range(merge):
                        coords.append((gh * merge + mh, gw * merge + mw))
    else:
        for h in range(grid_h):
            for w in range(grid_w):
                coords.append((h, w))
    N = len(coords)
    cos = torch.zeros(N, rdim); sin = torch.zeros(N, rdim)
    for i, (h, w) in enumerate(coords):
        ah = h * inv_freq; aw = w * inv_freq
        cos[i, :half] = torch.cos(ah); sin[i, :half] = torch.sin(ah)
        cos[i, half:] = torch.cos(aw); sin[i, half:] = torch.sin(aw)
    return cos.unsqueeze(0), sin.unsqueeze(0)


if __name__ == "__main__":
    print("=== PaddleOCR-VL modules ===")
    vis, emb, dec, head = VisionModel(), TextEmbed(), Decoder(), LMHead()
    for n, m in [("Vision", vis), ("Embed", emb), ("Decoder", dec), ("LMHead", head)]:
        print(f"  {n}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    gh, gw = 4, 4; N = gh * gw
    strip = torch.randn(1, 3, 14, 14 * N)
    pe = torch.randn(1, N, 1152)
    vc, vs = build_vision_rope_cos_sin(gh, gw)
    print("  Vision out:", list(vis(strip, pe, vc, vs).shape), "(expect [4,1024])")
    L = 8
    e = torch.randn(1, L, 1024); mask = build_causal_mask(L)
    pos = torch.arange(L).unsqueeze(0).repeat(3, 1)
    tc, ts = build_text_mrope_cos_sin(pos)
    print("  Decoder out:", list(dec(e, mask, tc, ts).shape), "(expect [1,8,1024])")
    print("  LMHead out:", list(head(dec(e, mask, tc, ts)[:, -1:, :]).shape), "(expect [1,1,103424])")
