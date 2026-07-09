"""End-to-end HunyuanOCR inference using ONLY the 4 exported TorchScript modules.

No KV cache: autoregression is simulated by re-running the full sequence each step
with a causal additive mask. The HF model is NOT used for inference here; the processor
is only used for tokenization / image preprocessing / position_ids construction.

Usage: python run_ts_ocr.py [image] [prompt] [max_new_tokens]
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
from transformers import AutoProcessor
import modules_modify as M

MODEL = "tencent/HunyuanOCR"
TS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts")
IMAGE_TOKEN = 120120
EOS = {120007, 120020}
DEFAULT_PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"


@torch.no_grad()
def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] image={img_path} device={dev}")

    proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
    vis = torch.jit.load(os.path.join(TS, "vision_encoder.pt"), map_location=dev).eval()
    emb = torch.jit.load(os.path.join(TS, "text_embed.pt"), map_location=dev).eval()
    dec = torch.jit.load(os.path.join(TS, "decoder.pt"), map_location=dev).eval()
    head = torch.jit.load(os.path.join(TS, "lm_head.pt"), map_location=dev).eval()

    # ---- preprocess via processor ----
    image = Image.open(img_path).convert("RGB")
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "image", "image": img_path},
                                          {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=image, padding=True, return_tensors="pt")
    input_ids = inp["input_ids"].to(dev)                 # (1,L)
    position_ids = inp["position_ids"].to(dev)           # (1,4,L)
    _, gh, gw = inp["image_grid_thw"][0].tolist()
    pv = inp["pixel_values"].float()

    # ---- vision + text embed + inject ----
    pixels = M.image_from_pixel_values(pv, gh, gw).to(dev)
    vtok = vis(pixels)                                   # (1,Lv,1024)
    embeds = emb(input_ids)                              # (1,L,1024)
    img_mask = (input_ids[0] == IMAGE_TOKEN)
    assert int(img_mask.sum()) == vtok.shape[1], (int(img_mask.sum()), vtok.shape[1])
    embeds[0][img_mask] = vtok[0].to(embeds.dtype)
    inv_freq = M.build_inv_freq().to(dev)

    print(f"[info] prefill L={input_ids.shape[1]}, generating (no kv-cache) ...")
    generated = []
    for step in range(max_new):
        L = embeds.shape[1]
        cos, sin = M.build_cos_sin(position_ids, inv_freq=inv_freq)
        mask = M.build_causal_mask(L).to(dev)
        hidden = dec(embeds, mask, cos, sin)             # (1,L,1024)
        logits = head(hidden[:, -1:, :])                 # (1,1,V)
        nxt = int(logits[0, -1].argmax().item())
        if nxt in EOS:
            break
        generated.append(nxt)
        # append token: grow embeds + position_ids (text token: all 4 axes = seq index L)
        nxt_t = torch.tensor([[nxt]], dtype=torch.long, device=dev)
        embeds = torch.cat([embeds, emb(nxt_t)], dim=1)
        newpos = torch.full((1, 4, 1), L, dtype=position_ids.dtype, device=dev)
        position_ids = torch.cat([position_ids, newpos], dim=2)
        if (step + 1) % 32 == 0:
            print(f"  ... {step + 1} tokens")

    result = proc.tokenizer.decode(generated, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
    print("\n===== OCR RESULT (TorchScript modules) =====")
    print(result)
    with open("ocr_result_ts.txt", "w", encoding="utf-8") as f:
        f.write(result)

    ref_path = "ocr_result.txt"
    if os.path.exists(ref_path):
        ref = open(ref_path, encoding="utf-8").read()
        print("\n===== match vs original run_ocr.py =====")
        print("IDENTICAL" if ref.strip() == result.strip() else "DIFF\n[original]\n" + ref)


if __name__ == "__main__":
    main()
