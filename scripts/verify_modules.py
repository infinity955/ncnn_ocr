"""Verify exported TorchScript modules against the HF model, per-module, on the real test image.
Also checks dynamic shapes and that SDPA/RoPE/GQA ops are present in the scripted graphs.
"""
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
import os
import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
import hyocr_modules_modify as M

MODEL = "d:/MySystem/share/SummerNcnn/pnnx/hunyuanocrmodel"
TS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts")
IMG = "../hunyuanocr_ncnn/assets/testimg.jpg"
PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"
IMAGE_TOKEN = 120120


def diff(a, b):
    a = a.float(); b = b.float()
    return (a - b).abs().max().item()


@torch.no_grad()
def main():
    proc = AutoProcessor.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    hf = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()

    vis = torch.jit.load(os.path.join(TS, "vision_encoder.pt")).eval()
    emb = torch.jit.load(os.path.join(TS, "text_embed.pt")).eval()
    dec = torch.jit.load(os.path.join(TS, "decoder.pt")).eval()
    head = torch.jit.load(os.path.join(TS, "lm_head.pt")).eval()

    # ---- processor inputs from real image ----
    image = Image.open(IMG).convert("RGB")
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "image", "image": IMG},
                                          {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=image, padding=True, return_tensors="pt")
    input_ids = inp["input_ids"]
    position_ids = inp["position_ids"]                  # (1,4,L)
    grid = inp["image_grid_thw"][0].tolist()            # [t,gh,gw]
    _, gh, gw = grid
    pv = inp["pixel_values"].float()
    L = input_ids.shape[1]
    print(f"[info] L={L}, grid={grid}")

    print("\n===== per-module vs HF =====")
    # 1) vision
    pixels = M.image_from_pixel_values(pv, gh, gw)      # (1,3,H,W)
    my_vis = vis(pixels)
    hf_vis = hf.vit(pv, inp["image_grid_thw"])
    print(f"[vision] my={tuple(my_vis.shape)} hf={tuple(hf_vis.shape)} maxabs={diff(my_vis, hf_vis):.3e}")

    # 2) embed
    my_emb = emb(input_ids)
    hf_emb = hf.model.embed_tokens(input_ids)
    print(f"[embed ] maxabs={diff(my_emb, hf_emb):.3e}")

    # build multimodal embeds (inject vision tokens)
    embeds = my_emb.clone()
    img_mask = (input_ids[0] == IMAGE_TOKEN)
    assert int(img_mask.sum()) == my_vis.shape[1], (int(img_mask.sum()), my_vis.shape[1])
    embeds[0][img_mask] = my_vis[0].to(embeds.dtype)

    # 3) decoder (validates rope cos/sin + causal mask + 24 layers + final norm)
    cos, sin = M.build_cos_sin(position_ids)
    mask = M.build_causal_mask(L)
    my_dec = dec(embeds, mask, cos, sin)
    hf_out = hf.model(inputs_embeds=embeds, position_ids=position_ids,
                      attention_mask=inp["attention_mask"], use_cache=False).last_hidden_state
    print(f"[decode] my={tuple(my_dec.shape)} hf={tuple(hf_out.shape)} maxabs={diff(my_dec, hf_out):.3e}")

    # 4) lm head
    my_logits = head(my_dec[:, -1:, :])
    hf_logits = hf.lm_head(hf_out[:, -1:, :])
    print(f"[lmhead] maxabs={diff(my_logits, hf_logits):.3e}")
    print(f"[argmax] my={my_logits[0, -1].argmax().item()} hf={hf_logits[0, -1].argmax().item()}")

    # ---- dynamic shape checks ----
    print("\n===== dynamic checks =====")
    for (h, w) in [(64, 96), (96, 64), (128, 128)]:
        o = vis(torch.randn(1, 3, h, w))
        print(f"[vision dyn] {h}x{w} -> {tuple(o.shape)}")
    for n in [5, 37, 128]:
        c, s = M.build_cos_sin(torch.arange(n).reshape(1, 1, n).repeat(1, 4, 1))
        o = dec(torch.randn(1, n, 1024), M.build_causal_mask(n), c, s)
        print(f"[decode dyn] len={n} -> {tuple(o.shape)}")

    # ---- graph structure checks ----
    print("\n===== graph op presence =====")
    for name in ["vision_encoder", "decoder"]:
        g = str(torch.jit.load(os.path.join(TS, name + ".pt")).inlined_graph)
        has_sdpa = "scaled_dot_product_attention" in g
        has_rope = ("aten::neg" in g) and ("aten::cat" in g)
        has_gqa = "repeat_interleave" in g
        print(f"[{name}] sdpa={has_sdpa} rope(neg+cat)={has_rope} repeat_interleave={has_gqa}")

    print("\n[done]")


if __name__ == "__main__":
    main()
