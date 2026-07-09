#!/usr/bin/env python3
"""DEBUG: compare PaddleOCR-VL vision features for C++ comparison."""
import sys, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
M = sys.argv[1]; img_path = sys.argv[2]
hf = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu").eval()

im = Image.open(img_path).convert("RGB")
w, h = im.size
eff = 28; round_by = lambda sz: max(eff, int(round(sz / eff) * eff))
scale = 1.0; area = h * w
if area > 1003520: scale = (1003520 / area) ** 0.5
elif area < 101920: scale = (101920 / area) ** 0.5
th, tw = round_by(h * scale), round_by(w * scale)
im = im.resize((tw, th), Image.BICUBIC)
arr = np.array(im).astype(np.float32) / 255.0
mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
arr = (arr - mean) / std

P, M_ = 14, 2; nph, npw = th // P, tw // P; N = nph * npw; gh, gw = nph // M_, npw // M_

@torch.no_grad()
def go():
    # patches (N,3,P,P) block-major & strip
    patches_np = np.zeros((N, 3, P, P), np.float32)
    idx = 0
    for bh in range(gh):
        for bw in range(gw):
            for mh in range(M_):
                for mw in range(M_):
                    h = bh * M_ + mh; w = bw * M_ + mw
                    patches_np[idx] = arr[h*P:(h+1)*P, w*P:(w+1)*P].transpose(2, 0, 1); idx += 1
    strip = torch.from_numpy(patches_np.transpose(1, 2, 0, 3).reshape(1, 3, P, P * N))

    # pos_embed: bilinear-interp 27x27 -> (nph,npw) raster, then reorder block-major
    pe_base = hf.visual.vision_model.embeddings.position_embedding.weight.float().detach()  # (729,1152)
    pe = pe_base.reshape(1, 27, 27, 1152).permute(0, 3, 1, 2)
    pe = torch.nn.functional.interpolate(pe, size=(nph, npw), mode="bilinear", align_corners=False)
    pe = pe.permute(0, 2, 3, 1).reshape(N, 1152)
    pe_bm = torch.zeros(N, 1152)
    idx = 0
    for bh in range(gh):
        for bw in range(gw):
            for mh in range(M_):
                for mw in range(M_):
                    h = bh * M_ + mh; w = bw * M_ + mw; pe_bm[idx] = pe[h * npw + w]; idx += 1
    pe_t = pe_bm.unsqueeze(0)

    # 2D rope block-major
    rdim, half = 36, 18; theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rdim))
    vcos = np.zeros((N, rdim), np.float32); vsin = np.zeros((N, rdim), np.float32)
    idx = 0
    for bh in range(gh):
        for gw_ in range(gw):
            for mh in range(M_):
                for mw in range(M_):
                    cur_h = bh * M_ + mh; cur_w = gw_ * M_ + mw
                    ah = cur_h * inv_freq.numpy(); aw = cur_w * inv_freq.numpy()
                    np.cos(ah, out=vcos[idx, :half]); np.sin(ah, out=vsin[idx, :half])
                    np.cos(aw, out=vcos[idx, half:]); np.sin(aw, out=vsin[idx, half:]); idx += 1
    vc_t = torch.from_numpy(vcos).unsqueeze(0); vs_t = torch.from_numpy(vsin).unsqueeze(0)

    # HF: raster patches -> visual + mlp_AR
    pv = patches_np.reshape(1, N, 3, P, P)
    pv_t = torch.from_numpy(pv)
    grid = [(1, nph, npw)]
    sig_pos = torch.arange(N) % (nph * npw)
    cu = torch.tensor([0, N], dtype=torch.int32); samp = torch.zeros(N, dtype=torch.int64)
    vout = hf.visual(pixel_values=pv_t, image_grid_thw=grid, position_ids=sig_pos,
                     vision_return_embed_list=True, interpolate_pos_encoding=True,
                     sample_indices=samp, cu_seqlens=cu, return_pooler_output=False,
                     use_rope=True, window_size=-1)
    hf_out = torch.cat(hf.mlp_AR(vout.last_hidden_state, grid), 0)  # (N/4, 1024) raster-merged

    # clean module (block-major input -> block-major-merged output = raster-merged order)
    sys.path.insert(0, HERE)
    from modules import VisionModel
    vis = VisionModel(); vis.load_state_dict(torch.load(os.path.join(HERE, "ts", "vision.pt"))); vis.eval()
    my_out = vis(strip, pe_t, vc_t, vs_t)  # (N/4, 1024)

    print(f"grid: {nph}x{npw}  N={N}  merged={N//4}")
    print(f"HF vision shape: {tuple(hf_out.shape)}  my module shape: {tuple(my_out.shape)}")
    print(f"HF vs my-module maxabs: {(hf_out - my_out).abs().max().item():.2e}")
    np.array([nph, npw, N], np.int32).tofile(os.path.join(HERE, "debug_info.bin"))
    hf_out.numpy().astype(np.float32).tofile(os.path.join(HERE, "hf_vision.bin"))
    my_out.numpy().astype(np.float32).tofile(os.path.join(HERE, "my_vision.bin"))
    print("dumps saved")

go()
