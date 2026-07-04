"""End-to-end HunyuanOCR inference using ONLY the 4 ncnn models (no torch modules).

torch/processor are used only for tokenization, image preprocessing and the cos/sin +
mask helpers. Inference (vision, embed, decoder, lm_head) runs entirely on ncnn.

ncnn conventions used throughout:
  * feed batch-stripped tensors (drop the leading 1);
  * text_embed indices are int32;
  * every input is np.ascontiguousarray;
  * fp16 arithmetic/storage disabled for fp32 accuracy;
  * no KV cache -> re-run the full sequence each step with a causal additive mask.
"""
import os
import sys
import numpy as np
import torch
import ncnn
from PIL import Image
from transformers import AutoProcessor
import hyocr_modules_modify as M

MODEL = "tencent/HunyuanOCR"
NC = os.path.dirname(os.path.abspath(__file__)) + "/ncnn"
IMAGE_TOKEN = 120120
EOS = {120007, 120020}
DEFAULT_PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"


def make_net(name):
    net = ncnn.Net()
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    assert net.load_param(f"{NC}/{name}.ncnn.param") == 0, name
    assert net.load_model(f"{NC}/{name}.ncnn.bin") == 0, name
    return net


def run(net, inputs):
    ex = net.create_extractor()
    for nm, arr in inputs:
        ex.input(nm, ncnn.Mat(np.ascontiguousarray(arr)).clone())
    _, out = ex.extract("out0")
    return np.array(out)


@torch.no_grad()
def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
    image = Image.open(img_path).convert("RGB")
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "image", "image": img_path},
                                          {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=image, padding=True, return_tensors="pt")
    input_ids = inp["input_ids"][0].numpy().astype(np.int64)     # (L0,)
    position_ids = inp["position_ids"]                           # (1,4,L0)
    _, gh, gw = inp["image_grid_thw"][0].tolist()
    pixels = M.image_from_pixel_values(inp["pixel_values"].float(), gh, gw)[0].numpy()  # (3,H,W)
    inv_freq = M.build_inv_freq()
    print(f"[info] image={img_path} L0={input_ids.shape[0]} grid=({gh},{gw})")

    print("[info] loading ncnn models ...")
    vis, emb, dec, head = (make_net(n) for n in ("vision_encoder", "text_embed", "decoder", "lm_head"))

    # vision + text embed + inject
    vtok = run(vis, [("in0", pixels)])                           # (Lv,1024)
    embeds = run(emb, [("in0", input_ids.astype(np.int32))])     # (L0,1024)
    img_mask = (input_ids == IMAGE_TOKEN)
    assert int(img_mask.sum()) == vtok.shape[0], (int(img_mask.sum()), vtok.shape[0])
    embeds[img_mask] = vtok
    print(f"[info] prefill L={embeds.shape[0]}, generating (ncnn, no kv-cache) ...")

    generated = []
    for step in range(max_new):
        L = embeds.shape[0]
        cos, sin = M.build_cos_sin(position_ids, inv_freq=inv_freq)   # (1,L,64)
        mask = M.build_causal_mask(L)[0].numpy()                     # (1,L,L)
        hidden = run(dec, [("in0", embeds), ("in1", mask),
                           ("in2", cos[0].numpy()), ("in3", sin[0].numpy())])  # (L,1024)
        logits = run(head, [("in0", hidden[-1:, :])])                # (1,120818)
        nxt = int(np.asarray(logits).reshape(-1).argmax())
        if nxt in EOS:
            break
        generated.append(nxt)
        emb_next = run(emb, [("in0", np.array([nxt], dtype=np.int32))]).reshape(1, -1)  # (1,1024)
        embeds = np.concatenate([embeds, emb_next], axis=0)
        newpos = torch.full((1, 4, 1), L, dtype=position_ids.dtype)
        position_ids = torch.cat([position_ids, newpos], dim=2)
        if (step + 1) % 16 == 0:
            print(f"  ... {step + 1} tokens")

    result = proc.tokenizer.decode(generated, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
    print("\n===== OCR RESULT (pure ncnn) =====")
    print(result)
    with open("ocr_result_ncnn.txt", "w", encoding="utf-8") as f:
        f.write(result)
    if os.path.exists("ocr_result.txt"):
        ref = open("ocr_result.txt", encoding="utf-8").read().strip()
        print("\n===== vs original run_ocr.py =====")
        print("IDENTICAL" if ref == result.strip() else "[original]\n" + ref)
    sys.stdout.flush()
    os._exit(0)   # skip ncnn allocator teardown crash on Windows


if __name__ == "__main__":
    main()
