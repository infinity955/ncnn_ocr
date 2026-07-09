#!/usr/bin/env python3
"""Generate PyTorch golden logits for C++ comparison using the same token sequence."""
import sys,os
sys.path.insert(0,'scripts/paddleocr_vl')
import numpy as np,torch
from PIL import Image
from transformers import AutoModelForCausalLM

M = sys.argv[1]; img_path = sys.argv[2]
hf = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu").eval()

im = Image.open(img_path).convert("RGB")
w,h = im.size; eff=28
area=h*w; scale=1.0
if area>1003520: scale=(1003520/area)**0.5
elif area<101920: scale=(101920/area)**0.5
th=int(round(h*scale/eff)*eff); tw=int(round(w*scale/eff)*eff)
im=im.resize((tw,th),Image.BICUBIC)
arr=np.array(im).astype(np.float32)/255.0
arr=(arr-np.array([0.48145466,0.4578275,0.40821073]))/np.array([0.26862954,0.26130258,0.27577711])
P=14; M2=2; nph,npw=th//P,tw//P; N=nph*npw; gh,gw=nph//M2,npw//M2

# Build raster patches + HF vision (like debug_vision.py)
raster=np.zeros((N,3,P,P),np.float32)
for hh in range(nph):
    for ww in range(npw):raster[hh*npw+ww]=arr[hh*P:(hh+1)*P,ww*P:(ww+1)*P].transpose(2,0,1)

with torch.no_grad():
    pv_t=torch.from_numpy(raster.reshape(1,N,3,P,P))
    cu=torch.tensor([0,N],dtype=torch.int32);samp=torch.zeros(N,dtype=torch.int64);sig=torch.arange(N)%(nph*npw)
    vout=hf.visual(pixel_values=pv_t,image_grid_thw=[(1,nph,npw)],position_ids=sig,
                   vision_return_embed_list=True,interpolate_pos_encoding=True,
                   sample_indices=samp,cu_seqlens=cu,return_pooler_output=False,use_rope=True,window_size=-1)
    vis_feat=torch.cat(hf.mlp_AR(vout.last_hidden_state,[(1,nph,npw)]),0)  # (350,1024)

# Same prompt tokens as C++
token_ids=[100273,8933,93963,101305]+[100295]*350+[101306,491,2497,93963,38606,93963]
print(f"seq={len(token_ids)} image_tokens=350")
ids_t=torch.tensor([token_ids])

# HF full prefill (bypass model.forward, use manual loop like verify)
from modules import build_causal_mask
cmask=build_causal_mask(len(token_ids))
# Use HF's actual layers copy
with torch.no_grad():
    emb=hf.model.embed_tokens(ids_t)
    # Inject image features
    img_mask=(ids_t==100295).squeeze(0)
    emb[0,img_mask]=vis_feat
    # Position IDs: build from hf.get_rope_index logic
    pos_type=ids_t[0,0]==100273  # check if we have vision tokens
    # Manual 3D position_ids: text tokens monotonically, image tokens (1, h, w) grid
    seq_len=len(token_ids)
    # Find image region
    im_start_idx=4  # after BOS User: IMAGE_START
    im_end_idx=im_start_idx+350  # right before IMAGE_END
    # Build position_ids: text before image = 0..3, image = (1,h,w), text after = 5..
    pos_ids=torch.zeros(3,1,seq_len,dtype=torch.long)
    # Text before image
    for i in range(im_start_idx):
        pos_ids[0,0,i]=i; pos_ids[1,0,i]=i; pos_ids[2,0,i]=i
    # Image tokens (350 patches, grid 14x25 after spatial merge /2)
    for i in range(350):
        hh=(i//25)//2; ww=(i%25)//2  # spatial_merge=2 -> merged grid
        t_idx=1; h_idx=hh; w_idx=ww
        pos_ids[0,0,im_start_idx+i]=t_idx; pos_ids[1,0,im_start_idx+i]=h_idx; pos_ids[2,0,im_start_idx+i]=w_idx
    # Text after image
    for i in range(im_start_idx+350,seq_len):
        p=i-350  # remove image token count from position
        pos_ids[0,0,i]=p; pos_ids[1,0,i]=p; pos_ids[2,0,i]=p
    # Run decoder manually (same as verify)
    pos_emb=hf.model.rotary_emb(ids_t.float(),pos_ids)
    hs=emb
    for layer in hf.model.layers:
        out=layer(hs,attention_mask=cmask,position_ids=pos_ids,position_embeddings=pos_emb)
        hs=out[0] if isinstance(out,(tuple,list)) else out
    hs=hf.model.norm(hs)
    logits=hf.lm_head(hs[:,-1:,:])  # (1,1,103424)
    print(f"logits shape: {tuple(logits.shape)}")
    top5=logits[0,0].argsort(descending=True)[:5]
    print(f"HF top5 tokens: {top5.tolist()}  values: {logits[0,0,top5].tolist()}")
    # Build next_token_id (argmax)
    next_tok=int(logits[0,0].argmax())
    print(f"HF next_token={next_tok}")
    logits[0,0].numpy().astype(np.float32).tofile(os.path.join(os.path.dirname(__file__),'hf_logits.bin'))


go()
