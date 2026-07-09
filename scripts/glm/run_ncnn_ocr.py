#!/usr/bin/env python3
"""
End-to-end GLM-OCR inference using pure ncnn models.

Uses the 4 ncnn modules (no PyTorch for inference) to run OCR on an image.
No KV cache — full sequence re-run each step.

Usage:
    python run_ncnn_ocr.py --image test.png
    python run_ncnn_ocr.py --model ./glm_ocr_model --ncnn ncnn --image test.png
"""

import argparse
import os
import sys

import numpy as np

# Path anchors (independent of cwd)
HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/glm
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # pnnx root


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR ncnn inference")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to HF model directory (for tokenizer/processor)")
    parser.add_argument("--ncnn", type=str, default=os.path.join(HERE, "ncnn"),
                        help="Path to ncnn model directory")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--prompt", "-p", type=str, default="Text Recognition:",
                        help="OCR prompt")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens to generate")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output text file (optional)")
    args = parser.parse_args()

    # Find model path
    model_path = args.model
    if model_path is None:
        candidate = os.path.join(REPO, "glm_ocr_model", "ZhipuAI", "glm-ocr")
        if os.path.isdir(candidate):
            model_path = candidate
    if model_path is None:
        home = os.path.expanduser("~")
        candidate = os.path.join(home, ".cache", "modelscope", "hub", "models", "ZhipuAI", "glm-ocr")
        if os.path.isdir(candidate):
            model_path = candidate

    if model_path is None or not os.path.exists(model_path):
        print("ERROR: Cannot find GLM-OCR model directory.")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"ncnn: {args.ncnn}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")

    # Load processor for tokenization and image preprocessing
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Load image
    from PIL import Image
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    # Build chat messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].numpy().astype(np.int32)
    pixel_values = inputs.get("pixel_values")
    print(f"Input tokens: {input_ids.shape[1]}")

    # Load ncnn models
    import ncnn
    nets = {}
    for name in ["vision", "text_embed", "decoder", "lm_head"]:
        param = os.path.join(args.ncnn, f"{name}.ncnn.param")
        bin_file = os.path.join(args.ncnn, f"{name}.ncnn.bin")
        if not os.path.exists(param):
            print(f"WARNING: {param} not found, skipping {name}")
            continue
        net = ncnn.Net()
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        net.load_param(param)
        net.load_model(bin_file)
        nets[name] = net
        print(f"Loaded {name}: {os.path.basename(param)}")

    def run_net(name: str, inputs: dict) -> np.ndarray:
        """Run ncnn net with named inputs, return output as numpy array."""
        ex = nets[name].create_extractor()
        for in_name, data in inputs.items():
            if data.dtype == np.int32:
                mat = ncnn.Mat(data.ravel())
            else:
                mat = ncnn.Mat(data.ravel(), data.shape[1], data.shape[0], data.shape[2] if data.ndim > 2 else 1)
            ex.input(in_name, mat)
        out = ncnn.Mat()
        ex.extract("out0", out)
        return np.array(out)

    # ============================================================
    # Vision Encoder
    # ============================================================
    if "vision" in nets and pixel_values is not None:
        print("\n--- Vision Encoder ---")
        # pixel_values from processor: (1, 3, H, W)
        pv = pixel_values.numpy().astype(np.float32)
        H, W = pv.shape[2], pv.shape[3]

        # Convert to image strip format
        from modules import image_to_strip
        pv_tensor = torch.from_numpy(pv)
        strip = image_to_strip(pv_tensor, patch_size=14, spatial_merge_size=2).numpy()

        # Build vision RoPE
        patch_size = 14
        spatial_merge = 2
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        N = num_patches_h * num_patches_w

        from modules import build_vision_rope_cos_sin
        cos, sin = build_vision_rope_cos_sin(
            num_patches_h, num_patches_w, spatial_merge, head_dim=64,
            mrope_section=[32, 32], rope_theta=10000.0,
        )
        cos_np = cos.numpy().astype(np.float32)
        sin_np = sin.numpy().astype(np.float32)

        vis_out = run_net("vision", {
            "in0": strip[0],  # (3, 14, 14*N) → strip batch dim
            "in1": cos_np[0],  # (N, 64)
            "in2": sin_np[0],
        })
        num_vision_tokens = vis_out.shape[0]
        print(f"  Vision tokens: {num_vision_tokens}, shape: {vis_out.shape}")
    else:
        print("WARNING: Vision encoder not available, using dummy vision features")
        num_vision_tokens = 64
        vis_out = np.zeros((num_vision_tokens, 1536), dtype=np.float32)

    # ============================================================
    # Build prompt with image tokens
    # ============================================================
    image_token_id = 59280
    full_prompt = "[gMASK]<sop><|user|>\n<|begin_of_image|>"
    for _ in range(num_vision_tokens):
        full_prompt += "<|image|>"
    full_prompt += "<|end_of_image|>" + args.prompt
    full_prompt += "/nothink<|assistant|>\n thinking"

    # Tokenize the full prompt
    if hasattr(tokenizer, 'encode'):
        token_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    else:
        token_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)

    print(f"Prompt tokens: {len(token_ids)}")

    # ============================================================
    # Text Embed + Vision Injection
    # ============================================================
    print("\n--- Text Embedding ---")
    token_ids_np = np.array(token_ids, dtype=np.int32)
    embeds = run_net("text_embed", {"in0": token_ids_np})
    print(f"  Embeddings shape: {embeds.shape}")

    # Inject vision features at image token positions
    image_positions = [i for i, tid in enumerate(token_ids) if tid == image_token_id]
    print(f"  Image token positions: {len(image_positions)}")
    if len(image_positions) == num_vision_tokens:
        for i, pos in enumerate(image_positions):
            embeds[pos] = vis_out[i]
    else:
        print(f"  WARNING: vision token mismatch! vision={num_vision_tokens}, image_tokens={len(image_positions)}")

    # ============================================================
    # Autoregressive generation (no KV cache)
    # ============================================================
    print(f"\n--- Generating (max {args.max_tokens} tokens) ---")
    eos_ids = {59246, 59253}  # EOS and EOP
    generated_ids = []
    seq_len = len(token_ids)

    import torch as _torch
    from modules import build_causal_mask, build_mrope_cos_sin

    for step in range(args.max_tokens):
        L = embeds.shape[0]

        # Build causal mask
        mask = build_causal_mask(L).numpy().astype(np.float32)

        # Build mRoPE position IDs: (1, 3, L)
        # For image tokens: 2D positions, for text tokens: 1D positions
        first_img = image_positions[0] if image_positions else L
        num_patches_w = 8  # approximate

        pos_t = _torch.arange(L, dtype=_torch.float32).unsqueeze(0)
        pos_h = _torch.zeros(1, L)
        pos_w = _torch.zeros(1, L)

        # Set 2D positions for image tokens
        for i, pos in enumerate(image_positions):
            h_idx = i // (num_patches_w // 2)
            w_idx = i % (num_patches_w // 2)
            pos_h[0, pos] = float(h_idx)
            pos_w[0, pos] = float(w_idx)

        # For text tokens after images, shift positions
        for i in range(L):
            if i >= first_img + num_vision_tokens:
                pos_t[0, i] = float(i - num_vision_tokens + num_patches_w // 2)
                pos_h[0, i] = 0.0
                pos_w[0, i] = 0.0

        pos_ids = _torch.stack([pos_t, pos_h, pos_w], dim=0).unsqueeze(0)  # (1, 3, L)
        cos, sin = build_mrope_cos_sin(pos_ids, head_dim=128,
                                       mrope_section=[16, 24, 24], rope_theta=10000.0)
        cos_np = cos.numpy().astype(np.float32)
        sin_np = sin.numpy().astype(np.float32)

        # Run decoder
        dec_out = run_net("decoder", {
            "in0": embeds.astype(np.float32),
            "in1": mask[0],  # (1, L, L)
            "in2": cos_np[0],  # (L, 64)
            "in3": sin_np[0],
        })
        # dec_out shape: (L, hidden)

        # Run LM head on last token
        last_hidden = dec_out[-1:].astype(np.float32)  # (1, hidden)
        logits = run_net("lm_head", {"in0": last_hidden})

        # Argmax
        next_token = int(np.argmax(logits))
        generated_ids.append(next_token)

        if next_token in eos_ids:
            break

        # Decode token for display
        if hasattr(tokenizer, 'decode'):
            token_text = tokenizer.decode([next_token], skip_special_tokens=False)
        else:
            token_text = str(next_token)
        print(token_text, end="", flush=True)

        # Append new token embedding
        next_emb = run_net("text_embed", {"in0": np.array([next_token], dtype=np.int32)})
        embeds = np.concatenate([embeds, next_emb], axis=0)

    print("\n\nDone.")

    # Decode full output
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True) if hasattr(tokenizer, 'decode') else " ".join(str(t) for t in generated_ids)
    print(f"\nOCR Result: {full_output}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_output)
        print(f"Saved to: {args.output}")

    # Avoid ncnn teardown crash on Windows
    os._exit(0)


if __name__ == "__main__":
    main()