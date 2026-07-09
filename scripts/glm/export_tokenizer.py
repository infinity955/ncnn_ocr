#!/usr/bin/env python3
"""
Export GLM-OCR tokenizer to ncnn-compatible format.

Reads the HF tokenizer from the model directory and outputs:
  - vocab.txt  (one token per line, line index == token ID)
  - merges.txt (BPE merge pairs)

Usage:
    python export_tokenizer.py
    python export_tokenizer.py --model ./glm_ocr_model --output ./assets/glm_ocr
"""

import argparse
import json
import os
import sys

# Path anchors (independent of cwd)
HERE = os.path.dirname(os.path.abspath(__file__))               # scripts/glm
NCNN_OCR = os.path.dirname(os.path.dirname(HERE))               # ncnn_ocr
REPO = os.path.dirname(NCNN_OCR)                                # pnnx root


def find_model_path():
    """Auto-detect model path."""
    candidate = os.path.join(REPO, "glm_ocr_model", "ZhipuAI", "glm-ocr")
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "tokenizer.json")):
        return candidate
    home = os.path.expanduser("~")
    cache_base = os.path.join(home, ".cache", "modelscope", "hub", "models", "ZhipuAI", "glm-ocr")
    if os.path.isdir(cache_base) and os.path.exists(os.path.join(cache_base, "tokenizer.json")):
        return cache_base
    return None


def main():
    parser = argparse.ArgumentParser(description="Export GLM-OCR tokenizer")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--output", "-o", type=str, default=os.path.join(NCNN_OCR, "assets", "glm_ocr"),
                        help="Output directory for tokenizer files")
    args = parser.parse_args()

    model_path = args.model or find_model_path()
    if model_path is None:
        print("ERROR: Cannot find GLM-OCR model.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer.json
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: tokenizer.json not found in {model_path}")
        sys.exit(1)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    model_data = tok_data.get("model", {})
    vocab = model_data.get("vocab", {})
    merges = model_data.get("merges", [])

    print(f"Vocab size: {len(vocab)}")
    print(f"Merge pairs: {len(merges)}")

    # Build id-to-token mapping
    max_id = max(vocab.values()) if vocab else 0

    # Added/special tokens (e.g. <|image|>=59280, <|begin_of_image|>, [gMASK]) live
    # separately from the base vocab in tokenizer.json — they MUST be included at
    # their IDs or the C++ can't insert image placeholders.
    added_tokens = tok_data.get("added_tokens", [])
    for t in added_tokens:
        tid = t.get("id", -1)
        if tid > max_id:
            max_id = tid

    # Pad to the model's vocab_size (lm_head width) so C++ vocab matches (unused slots
    # get placeholder tokens).
    try:
        cfg = json.load(open(os.path.join(model_path, "config.json"), encoding="utf-8"))
        vocab_size = cfg.get("text_config", {}).get("vocab_size", max_id + 1)
    except Exception:
        vocab_size = max_id + 1
    n = max(max_id + 1, vocab_size)

    id_to_token = [""] * n
    for token, tid in vocab.items():
        id_to_token[tid] = token
    for t in added_tokens:                       # place special tokens at their IDs
        tid = t.get("id", -1)
        if 0 <= tid < n:
            id_to_token[tid] = t.get("content", "")

    # Fill remaining gaps with unique placeholders
    for i in range(len(id_to_token)):
        if id_to_token[i] == "":
            id_to_token[i] = f"<|unused_{i}|>"

    # Write vocab.txt
    vocab_path = os.path.join(args.output, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in id_to_token:
            f.write(token + "\n")
    print(f"Wrote: {vocab_path} ({len(id_to_token)} lines)")

    # Write merges.txt (newer tokenizer.json stores merges as [a, b] pairs, older as "a b")
    merges_path = os.path.join(args.output, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            line = " ".join(merge) if isinstance(merge, (list, tuple)) else merge
            f.write(line + "\n")
    print(f"Wrote: {merges_path} ({len(merges)} lines)")

    # Also write special_tokens.json for reference
    special_tokens = {}
    for t in added_tokens:
        special_tokens[str(t.get("id", -1))] = {
            "token": t.get("content", ""),
            "id": t.get("id", -1),
            "special": t.get("special", False),
        }

    sp_path = os.path.join(args.output, "special_tokens.json")
    with open(sp_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {sp_path}")

    print(f"\nTokenizer files ready in: {args.output}/")


if __name__ == "__main__":
    main()