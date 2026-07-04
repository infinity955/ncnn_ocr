"""Clean, TorchScript-friendly reimplementation of HunyuanOCR split into 4 modules.

Hard constraints (ncnn target):
  * batch is always 1 (batch dim written as literal 1).
  * no intermediate tensor with ndim >= 5.

Modules:
  1. VisionEncoder : (1,3,H,W) -> (1,Lv,1024)   [conv patch-embed + 27 ViT blocks + patch merger]
  2. TextEmbed     : (1,len)   -> (1,len,1024)
  3. Decoder       : (1,len,1024),(1,1,len,len),(1,len,64),(1,len,64) -> (1,len,1024)  [24 layers + final RMSNorm]
  4. LMHead        : (1,len,1024) -> (1,len,120818)

The rotary cos/sin are computed OUTSIDE (see build_cos_sin) and fed as head_dim/2 = 64;
the decoder duplicates them to 128 internally (cat), matching the non-interleaved RoPE pattern.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers (norm / rope)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return self.weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


# ============================================================================= 1. VISION ENCODER
class VisionMLP(nn.Module):
    def __init__(self, hidden: int = 1152, inter: int = 4304):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden, inter, bias=True)
        self.dense_4h_to_h = nn.Linear(inter, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))


class VisionAttention(nn.Module):
    def __init__(self, hidden: int = 1152, num_heads: int = 16, head_dim: int = 72):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        q = self.q_proj(x).reshape(1, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, k, v)  # no mask, full attention
        o = o.permute(0, 2, 1, 3).reshape(1, n, self.num_heads * self.head_dim)
        return self.o_proj(o)


class VisionBlock(nn.Module):
    def __init__(self, hidden: int = 1152, inter: int = 4304, num_heads: int = 16,
                 head_dim: int = 72, eps: float = 1e-5):
        super().__init__()
        self.self_attn = VisionAttention(hidden, num_heads, head_dim)
        self.mlp = VisionMLP(hidden, inter)
        self.input_layernorm = nn.LayerNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class VisionPatchMerger(nn.Module):
    def __init__(self, in_ch: int = 1152, out_ch: int = 1024, merge: int = 2, eps: float = 1e-5):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=merge, stride=merge),
            nn.GELU(),
            nn.Conv2d(in_ch * 2, in_ch * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_ch * 4, out_ch)
        self.image_newline = nn.Parameter(torch.zeros(in_ch * 4))
        self.image_begin = nn.Parameter(torch.zeros(out_ch))
        self.image_end = nn.Parameter(torch.zeros(out_ch))
        self.before_rms = RMSNorm(in_ch, eps)
        self.after_rms = RMSNorm(out_ch, eps)

    def forward(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        x = self.before_rms(x)                                   # (1,Nv,1152)
        x = x.permute(0, 2, 1).reshape(1, self.in_ch, gh, gw)    # (1,1152,gh,gw)
        x = self.proj(x)                                         # (1,4608,gh/2,gw/2)
        c = x.size(1)
        # newline column via broadcast-add-zero (avoids dynamic expand -> pnnx.Expression,
        # which ncnn can't run). nl(1,c,1,1) + zeros(1,1,h2,1) -> (1,c,h2,1), each row = nl.
        newline = self.image_newline.reshape(1, c, 1, 1) + x[:, :1, :, :1] * 0.0
        x = torch.cat([x, newline], dim=3)                       # (1,4608,gh/2,gw/2+1)
        x = x.reshape(1, c, -1).permute(0, 2, 1)                 # (1,L,4608)
        x = self.mlp(x)                                          # (1,L,1024)
        begin = self.image_begin.reshape(1, 1, self.out_ch)
        end = self.image_end.reshape(1, 1, self.out_ch)
        x = torch.cat([begin, x, end], dim=1)                    # (1,L+2,1024)
        return self.after_rms(x)


class VisionEncoder(nn.Module):
    def __init__(self, hidden: int = 1152, inter: int = 4304, num_layers: int = 27,
                 num_heads: int = 16, head_dim: int = 72, out_ch: int = 1024,
                 patch: int = 16, merge: int = 2, pos_edge: int = 128, eps: float = 1e-5):
        super().__init__()
        self.hidden = hidden
        self.pos_edge = pos_edge
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch, bias=True)
        self.layers = nn.ModuleList(
            [VisionBlock(hidden, inter, num_heads, head_dim, eps) for _ in range(num_layers)]
        )
        self.perceive = VisionPatchMerger(hidden, out_ch, merge, eps)

    def forward(self, pixels: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(pixels)                         # (1,1152,gh,gw)
        gh = x.size(2)
        gw = x.size(3)
        x = x + pos_embed                                       # (1,1152,gh,gw) — pre-computed externally
        x = x.reshape(1, self.hidden, gh * gw).permute(0, 2, 1)  # (1,Nv,1152)
        for layer in self.layers:
            x = layer(x)
        return self.perceive(x, gh, gw)                          # (1,Lv,1024)


# ============================================================================= 2. TEXT EMBED
class TextEmbed(nn.Module):
    def __init__(self, vocab: int = 120818, hidden: int = 1024):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


# ============================================================================= 3. DECODER
class DecoderAttention(nn.Module):
    def __init__(self, hidden: int = 1024, n_q: int = 16, n_kv: int = 8,
                 head_dim: int = 128, eps: float = 1e-5):
        super().__init__()
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.rep = n_q // n_kv
        self.q_proj = nn.Linear(hidden, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, hidden, bias=False)
        self.query_layernorm = RMSNorm(head_dim, eps)
        self.key_layernorm = RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        q = self.q_proj(x).reshape(1, n, self.n_q, self.head_dim).permute(0, 2, 1, 3)   # (1,16,n,128)
        k = self.k_proj(x).reshape(1, n, self.n_kv, self.head_dim).permute(0, 2, 1, 3)  # (1,8,n,128)
        v = self.v_proj(x).reshape(1, n, self.n_kv, self.head_dim).permute(0, 2, 1, 3)  # (1,8,n,128)
        # rope: duplicate head_dim/2 cos/sin to head_dim, then rotate_half.
        # keep cos/sin 3D (1,n,128) -- broadcasts over heads via right-alignment; an
        # explicit unsqueeze(1) makes pnnx lay the cache out transposed for ncnn RotaryEmbed.
        cos2 = torch.cat([cos, cos], dim=-1)  # (1,n,128)
        sin2 = torch.cat([sin, sin], dim=-1)
        q = q * cos2 + rotate_half(q) * sin2
        k = k * cos2 + rotate_half(k) * sin2
        # qk-norm (RMSNorm over head_dim)
        q = self.query_layernorm(q)
        k = self.key_layernorm(k)
        # GQA expand
        k = k.repeat_interleave(self.rep, dim=-3)  # (1,16,n,128)
        v = v.repeat_interleave(self.rep, dim=-3)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.permute(0, 2, 1, 3).reshape(1, n, self.n_q * self.head_dim)
        return self.o_proj(o)


class DecoderMLP(nn.Module):
    def __init__(self, hidden: int = 1024, inter: int = 3584):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden: int = 1024, inter: int = 3584, n_q: int = 16, n_kv: int = 8,
                 head_dim: int = 128, eps: float = 1e-5):
        super().__init__()
        self.self_attn = DecoderAttention(hidden, n_q, n_kv, head_dim, eps)
        self.mlp = DecoderMLP(hidden, inter)
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers: int = 24, hidden: int = 1024, inter: int = 3584,
                 n_q: int = 16, n_kv: int = 8, head_dim: int = 128, eps: float = 1e-5):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden, inter, n_q, n_kv, head_dim, eps) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden, eps)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = hidden
        for layer in self.layers:
            x = layer(x, mask, cos, sin)
        return self.norm(x)


# ============================================================================= 4. LM HEAD
class LMHead(nn.Module):
    def __init__(self, hidden: int = 1024, vocab: int = 120818):
        super().__init__()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)


# ============================================================================= driver-side helpers (NOT scripted; ndim>4 allowed here)
def build_inv_freq(head_dim: int = 128, rope_theta: float = 10000.0,
                   alpha: float = 1000.0) -> torch.Tensor:
    base = rope_theta * (alpha ** (head_dim / (head_dim - 2)))
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # (64,)


def build_cos_sin(position_ids: torch.Tensor, head_dim: int = 128,
                  section: int = 16, inv_freq: torch.Tensor = None):
    """position_ids: (1,4,L) int64 -> cos,sin: (1,L,head_dim/2) fp32.

    dim d in [0, head_dim/2): axis a = d // section (4 axes x 16 dims).
    cos_half[s,d] = cos(position_ids[0,a,s] * inv_freq[d]).
    """
    if inv_freq is None:
        inv_freq = build_inv_freq(head_dim)
    half = head_dim // 2
    pos = position_ids[0].to(torch.float32)          # (4,L)
    axis = torch.arange(half) // section             # (half,) values in 0..3
    pos_sel = pos[axis]                              # (half,L)
    angles = pos_sel.transpose(0, 1) * inv_freq.unsqueeze(0)  # (L,half)
    # .contiguous(): angles derives from a transpose; downstream consumers that hand the
    # raw buffer to ncnn.Mat need C-contiguous memory or the data is read transposed.
    cos = angles.cos().unsqueeze(0).contiguous()
    sin = angles.sin().unsqueeze(0).contiguous()
    return cos, sin


def build_causal_mask(length: int, dtype=torch.float32) -> torch.Tensor:
    """(1,1,length,length) additive causal mask: 0 where j<=i else large negative."""
    neg = torch.finfo(dtype).min
    m = torch.full((length, length), neg, dtype=dtype)
    m = torch.triu(m, diagonal=1)                    # keep upper (j>i) = neg, lower/diag = 0
    return m.reshape(1, 1, length, length)


def image_from_pixel_values(pixel_values: torch.Tensor, gh: int, gw: int,
                            patch: int = 16) -> torch.Tensor:
    """Reconstruct the normalized (1,3,H,W) image from processor pixel_values (Nv,768).

    Row-major over (gh,gw); each row is a patch flattened as (C,ph,pw).
    (Driver-side only; uses 5D reshape which is fine outside the exported modules.)
    """
    pv = pixel_values.reshape(gh, gw, 3, patch, patch)
    img = pv.permute(2, 0, 3, 1, 4).reshape(1, 3, gh * patch, gw * patch)
    return img
