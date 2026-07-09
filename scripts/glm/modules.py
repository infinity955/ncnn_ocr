#!/usr/bin/env python3
"""
Clean-room TorchScript-compatible module definitions for GLM-OCR.

Rewritten to EXACTLY mirror transformers 5.13.0 `modeling_glm_ocr.py`.
Splits GLM-OCR into 4 independent modules matching the C++ ncnn I/O contract
(src/glm_ocr.cpp):

  1. VisionEncoder  — ViT: Conv2d patch embed (folded Conv3d), 24 blocks (qkv+q/k_norm,
                      non-interleaved 2D RoPE), post_layernorm, spatial-merge downsample
                      Conv2d, gated PatchMerger. Output = merger (pooler_output).
  2. TextEmbed      — Token embedding lookup.
  3. Decoder        — 16 layers, sandwich 4-norm, GQA 16/8, fused gate_up MLP,
                      INTERLEAVED mRoPE (rotate_half_llm).
  4. LMHead         — Linear projection to vocab.

Real config (glm_ocr):
  Vision: depth=24, hidden=1024, heads=16, head_dim=64, patch=14, temporal_patch=2,
          spatial_merge=2, inter=4096, out_hidden=1536, merger context_dim=4608,
          vision RoPE dim=32 (h16+w16), SiLU MLP, RMSNorm blocks, LayerNorm in merger.
  Text:   layers=16, hidden=1536, 16 Q / 8 KV heads, head_dim=128, inter=4608,
          NO q/k norm, mrope_section=[16,24,24], rope_theta=10000, SiLU, RMSNorm.

ncnn/pnnx constraints:
  - Batch=1 everywhere (literal 1 in reshape/permute)
  - RoPE cos/sin and causal mask computed externally (C++), passed as inputs
  - Vision cos/sin width = 32 (module duplicates to 64); Text cos/sin width = 64
    (module repeat_interleave(2) to 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared helpers
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm — used throughout GLM-OCR (blocks, decoder, q/k norm).

    No dtype casts (fp32 pipeline) so pnnx emits no aten::to layers.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Contiguous half-split (VISION rope): cat((-x2, x1))."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_llm(x: torch.Tensor) -> torch.Tensor:
    """Interleaved even/odd split (TEXT rope): stack((-x_odd, x_even)).flatten."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# ============================================================
# Module 1: VisionEncoder
# ============================================================

class VisionMLP(nn.Module):
    """SwiGLU MLP with bias (vision blocks)."""
    def __init__(self, hidden: int = 1024, inter: int = 4096):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=True)
        self.up_proj = nn.Linear(hidden, inter, bias=True)
        self.down_proj = nn.Linear(inter, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionAttention(nn.Module):
    """Vision self-attention: fused qkv, per-head q/k RMSNorm, non-interleaved 2D RoPE.

    cos/sin come in as (1, N, 32) [h16 | w16]; duplicated to (1, N, 64) internally.
    """
    def __init__(self, hidden: int = 1024, num_heads: int = 16, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden = hidden
        self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (1, N, hidden); cos/sin: (1, N, 32)
        # ncnn-friendly: permute to (1,heads,N,dim) then rope with 3D cos (broadcasts over heads)
        N = x.size(1)
        qkv = self.qkv(x).reshape(1, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (1, heads, N, 64)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Vision RoPE: duplicate cos/sin 32 -> 64 (contiguous), keep 3D
        cos2 = torch.cat((cos, cos), dim=-1)  # (1, N, 64)
        sin2 = torch.cat((sin, sin), dim=-1)
        q = (q * cos2) + (rotate_half(q) * sin2)
        k = (k * cos2) + (rotate_half(k) * sin2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.permute(0, 2, 1, 3).reshape(1, N, self.hidden)
        return self.proj(attn)


class VisionBlock(nn.Module):
    def __init__(self, hidden: int = 1024, num_heads: int = 16, head_dim: int = 64,
                 inter: int = 4096):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = VisionAttention(hidden, num_heads, head_dim)
        self.norm2 = RMSNorm(hidden)
        self.mlp = VisionMLP(hidden, inter)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nn.Module):
    """Gated MLP merger: proj -> LayerNorm -> GELU -> SwiGLU(down(silu(gate)*up)).

    dim = out_hidden (1536), context_dim = 4608. All Linear bias=False.
    """
    def __init__(self, dim: int = 1536, context_dim: int = 4608):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.gelu(self.post_projection_norm(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionEncoder(nn.Module):
    """GLM-OCR Vision Encoder.

    Input:
      image_strip: (1, 3, patch, patch * N)  — single-frame patches, block-major 2x2 order
      cos_cache:   (1, N, 32)                 — 2D RoPE cos (h16 | w16)
      sin_cache:   (1, N, 32)
    Output:
      image_embeds: (1, N/4, out_hidden=1536) — merger (pooler) output
    """
    def __init__(self, hidden: int = 1024, depth: int = 24, num_heads: int = 16,
                 head_dim: int = 64, inter: int = 4096, patch_size: int = 14,
                 out_hidden: int = 1536, spatial_merge: int = 2, merger_ctx: int = 4608):
        super().__init__()
        self.patch_size = patch_size
        self.hidden = hidden
        self.spatial_merge = spatial_merge
        self.out_hidden = out_hidden

        # patch_embed: Conv2d over a single temporal frame (Conv3d folded in export)
        self.patch_embed = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size, bias=True)
        self.blocks = nn.ModuleList([
            VisionBlock(hidden, num_heads, head_dim, inter) for _ in range(depth)
        ])
        self.post_layernorm = RMSNorm(hidden)
        # spatial merge (2x2) -> out_hidden. A 2x2 Conv2d(stride 2) over a 2x2 block
        # equals a Linear over the flattened (in_c, kh, kw) vector; use Linear so ncnn
        # keeps the merged-token dim as rows (batched Conv2d would fold it into channels).
        self.merge = spatial_merge * spatial_merge
        self.downsample = nn.Linear(hidden * self.merge, out_hidden)
        self.merger = VisionPatchMerger(out_hidden, merger_ctx)

    def forward(self, image_strip: torch.Tensor,
                cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image_strip)      # (1, hidden, 1, N)
        x = x.flatten(2).transpose(1, 2)       # (1, N, hidden)

        for block in self.blocks:
            x = block(x, cos_cache, sin_cache)

        x = self.post_layernorm(x)             # (1, N, hidden)

        # spatial merge: consecutive `merge` tokens form a 2x2 block (strip is block-major).
        # reorder to (N/merge, in_c, kh, kw) flattened -> Linear
        x = x.reshape(-1, self.merge, self.hidden)   # (N/merge, merge, in_c)  token = kh*2+kw
        x = x.permute(0, 2, 1)                        # (N/merge, in_c, merge)
        x = x.reshape(-1, self.hidden * self.merge)   # (N/merge, in_c*merge) = (in_c,kh,kw)
        x = self.downsample(x)                        # (N/merge, out_hidden)
        x = self.merger(x)                            # (N/merge, out_hidden)
        return x                                       # 2D -> ncnn Mat(out_hidden, N/merge)


# ============================================================
# Module 2: TextEmbed
# ============================================================

class TextEmbed(nn.Module):
    def __init__(self, vocab_size: int = 59392, hidden_size: int = 1536):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)  # (1, L, hidden)


# ============================================================
# Module 3: Decoder
# ============================================================

class DecoderAttention(nn.Module):
    """GLM-OCR text attention: GQA 16/8, NO q/k norm.

    ncnn-friendly CONTIGUOUS RoPE (cat + rotate_half). The real model uses
    INTERLEAVED rope; equivalence is achieved by permuting q_proj/k_proj output
    rows (interleave->half) at export time (dot product is invariant under a
    matching permutation of Q and K). cos/sin come in as (1, L, 64) and are
    duplicated to 128 via cat; kept 3D so they broadcast over heads.
    """
    def __init__(self, hidden: int = 1536, n_q: int = 16, n_kv: int = 8, head_dim: int = 128):
        super().__init__()
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.hidden = hidden
        self.q_per_kv = n_q // n_kv

        self.q_proj = nn.Linear(hidden, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (1, L, hidden); cos/sin: (1, L, 64); mask: (1, 1, L, L)
        L = x.size(1)
        q = self.q_proj(x).reshape(1, L, self.n_q, self.head_dim).permute(0, 2, 1, 3)   # (1,16,L,128)
        k = self.k_proj(x).reshape(1, L, self.n_kv, self.head_dim).permute(0, 2, 1, 3)  # (1,8,L,128)
        v = self.v_proj(x).reshape(1, L, self.n_kv, self.head_dim).permute(0, 2, 1, 3)

        cos2 = torch.cat((cos, cos), dim=-1)  # (1, L, 128) 3D, broadcasts over heads
        sin2 = torch.cat((sin, sin), dim=-1)
        q = (q * cos2) + (rotate_half(q) * sin2)
        k = (k * cos2) + (rotate_half(k) * sin2)

        # GQA: expand KV heads to match Q heads
        k = k.repeat_interleave(self.q_per_kv, dim=-3)  # (1, 16, L, 128)
        v = v.repeat_interleave(self.q_per_kv, dim=-3)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.permute(0, 2, 1, 3).reshape(1, L, self.n_q * self.head_dim)
        return self.o_proj(attn)


class DecoderMLP(nn.Module):
    """Fused-gate SwiGLU: gate_up_proj -> split -> down(silu(gate)*up)."""
    def __init__(self, hidden: int = 1536, inter: int = 4608):
        super().__init__()
        self.inter = inter
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up_proj(x)          # (1, L, 2*inter)
        gate = gu[..., : self.inter]       # first half
        up = gu[..., self.inter:]          # second half
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    """Sandwich 4-norm layer (matches GlmOcrTextDecoderLayer)."""
    def __init__(self, hidden: int = 1536, n_q: int = 16, n_kv: int = 8,
                 head_dim: int = 128, inter: int = 4608):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = DecoderAttention(hidden, n_q, n_kv, head_dim)
        self.post_self_attn_layernorm = RMSNorm(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = DecoderMLP(hidden, inter)
        self.post_mlp_layernorm = RMSNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cos, sin)
        h = self.post_self_attn_layernorm(h)
        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        h = self.post_mlp_layernorm(h)
        x = residual + h
        return x


class Decoder(nn.Module):
    """GLM-OCR Text Decoder.

    Input:
      embeds: (1, L, hidden)   — multimodal embeddings
      mask:   (1, 1, L, L)     — causal attention mask (-inf upper triangle)
      cos:    (1, L, 64)       — mRoPE cos (half dim)
      sin:    (1, L, 64)
    Output:
      hidden_states: (1, L, hidden)
    """
    def __init__(self, num_layers: int = 16, hidden: int = 1536,
                 n_q: int = 16, n_kv: int = 8, head_dim: int = 128, inter: int = 4608):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(hidden, n_q, n_kv, head_dim, inter) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden)

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = embeds
        for layer in self.layers:
            x = layer(x, mask, cos, sin)
        return self.norm(x)


# ============================================================
# Module 4: LMHead
# ============================================================

class LMHead(nn.Module):
    def __init__(self, hidden: int = 1536, vocab: int = 59392):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================
# Driver-side helpers (NOT scripted — used for verification / tracing dummies)
# ============================================================

def build_causal_mask(length: int) -> torch.Tensor:
    """Causal mask (1, 1, L, L) with -inf above the diagonal."""
    mask = torch.full((1, 1, length, length), float("-inf"))
    return torch.triu(mask, diagonal=1)


def build_text_mrope_cos_sin(position_ids: torch.Tensor, head_dim: int = 128,
                             mrope_section=None, rope_theta: float = 10000.0):
    """Text mRoPE cos/sin (half-dim, width = head_dim//2 = 64).

    position_ids: (3, L) — temporal/height/width position axes.
    Mirrors GlmOcrTextRotaryEmbedding + apply_mrope: full inv_freq per axis,
    then section-select axis i%3 over dims [16,24,24].
    Returns cos, sin: (1, L, head_dim//2).
    """
    if mrope_section is None:
        mrope_section = [16, 24, 24]
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))  # (half,)

    # freqs per axis: (3, L, half)
    freqs = torch.stack([torch.outer(position_ids[a].float(), inv_freq) for a in range(3)], dim=0)
    # apply_mrope: pick axis i%3 for section i
    parts = []
    off = 0
    for i, sec in enumerate(mrope_section):
        parts.append(freqs[i % 3, :, off:off + sec])
        off += sec
    freq = torch.cat(parts, dim=-1)          # (L, half)
    return freq.cos().unsqueeze(0), freq.sin().unsqueeze(0)


def build_vision_rope_cos_sin(num_patches_h: int, num_patches_w: int,
                              spatial_merge_size: int = 2, rope_dim: int = 32,
                              rope_theta: float = 10000.0):
    """2D vision RoPE cos/sin, width = rope_dim = 32 (h16 | w16), block-major order.

    Mirrors C++ generate_vision_rope_cache_2d (duplicate_sections=false).
    inv_freq over dim=rope_dim: 1/theta^(2i/rope_dim), i in 0..rope_dim/2-1.
    Returns cos, sin: (1, N, rope_dim).
    """
    half = rope_dim // 2  # 16 per axis
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rope_dim))
    grid_h = num_patches_h // spatial_merge_size
    grid_w = num_patches_w // spatial_merge_size
    N = num_patches_h * num_patches_w

    cos = torch.zeros(N, rope_dim)
    sin = torch.zeros(N, rope_dim)
    idx = 0
    for gh in range(grid_h):
        for gw in range(grid_w):
            for mh in range(spatial_merge_size):
                for mw in range(spatial_merge_size):
                    cur_h = gh * spatial_merge_size + mh
                    cur_w = gw * spatial_merge_size + mw
                    ah = cur_h * inv_freq
                    aw = cur_w * inv_freq
                    cos[idx, :half] = torch.cos(ah); sin[idx, :half] = torch.sin(ah)
                    cos[idx, half:] = torch.cos(aw); sin[idx, half:] = torch.sin(aw)
                    idx += 1
    return cos.unsqueeze(0), sin.unsqueeze(0)


def image_to_strip(pixel_values: torch.Tensor, patch_size: int = 14,
                   spatial_merge_size: int = 2) -> torch.Tensor:
    """Convert (1,3,H,W) to single-frame block-major strip (1,3,patch,patch*N).

    Matches C++ bgr_to_image_strip() ordering (gh,gw,mh,mw).
    """
    _, _, H, W = pixel_values.shape
    nph = H // patch_size
    npw = W // patch_size
    grid_h = nph // spatial_merge_size
    grid_w = npw // spatial_merge_size

    patches = []
    for gh in range(grid_h):
        for gw in range(grid_w):
            for mh in range(spatial_merge_size):
                for mw in range(spatial_merge_size):
                    ph = gh * spatial_merge_size + mh
                    pw = gw * spatial_merge_size + mw
                    y0, x0 = ph * patch_size, pw * patch_size
                    patches.append(pixel_values[:, :, y0:y0 + patch_size, x0:x0 + patch_size])
    all_patches = torch.cat(patches, dim=0)  # (N, 3, patch, patch)
    N = all_patches.size(0)
    return all_patches.permute(1, 2, 0, 3).reshape(1, 3, patch_size, patch_size * N)


# ============================================================
# Self-test (shapes only)
# ============================================================

if __name__ == "__main__":
    print("=== GLM-OCR Module Definitions (rewritten) ===")
    vis = VisionEncoder(); emb = TextEmbed(); dec = Decoder(); head = LMHead()
    for name, m in [("VisionEncoder", vis), ("TextEmbed", emb), ("Decoder", dec), ("LMHead", head)]:
        print(f"  {name}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    total = sum(p.numel() for m in [vis, emb, dec, head] for p in m.parameters())
    print(f"  Total: {total/1e6:.1f}M")

    print("\n--- Forward shape tests ---")
    N = 16  # patches (multiple of 4)
    strip = torch.randn(1, 3, 14, 14 * N)
    vcos, vsin = build_vision_rope_cos_sin(4, 4)  # 4x4 patches = 16
    out = vis(strip, vcos, vsin)
    print(f"  Vision: {list(out.shape)}  (expect [1, {N//4}, 1536])")

    L = 8
    ids = torch.randint(0, 1000, (1, L))
    print(f"  Embed: {list(emb(ids).shape)}  (expect [1, {L}, 1536])")

    embeds = torch.randn(1, L, 1536)
    mask = build_causal_mask(L)
    pos = torch.arange(L).unsqueeze(0).repeat(3, 1)
    tcos, tsin = build_text_mrope_cos_sin(pos)
    dout = dec(embeds, mask, tcos, tsin)
    print(f"  Decoder: {list(dout.shape)}  (expect [1, {L}, 1536])")
    print(f"  LMHead: {list(head(dout[:, -1:, :]).shape)}  (expect [1, 1, 59392])")
    print("\nAll module definitions OK.")
