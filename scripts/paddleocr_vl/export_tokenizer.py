#!/usr/bin/env python3
"""
Export PaddleOCR-VL tokenizer to vocab.txt / merges.txt for the C++ runtime.

NOTE: PaddleOCR-VL uses a SentencePiece-style BPE (model.type=BPE, byte_fallback=True,
metaspace '▁' word boundary, decoder Replace('▁'->' ') + ByteFallback + Fuse).
This is DIFFERENT from GLM/HunYuan GPT-2 byte-level BBPE. The C++ decode must:
  - map token id -> token string
  - concatenate, replace '▁' -> ' '
  - byte-fallback: tokens like '<0xE4>' -> raw byte 0xE4, then fuse consecutive bytes -> UTF-8

Outputs vocab.txt (line index == id, padded to vocab_size) + merges.txt + special_tokens.json.

Usage:
  .venv-paddle/Scripts/python.exe scripts/paddleocr_vl/export_tokenizer.py --model <snapshot> --output assets/paddleocr_vl
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--output", "-o", default=os.path.join(ROOT, "assets", "paddleocr_vl"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    tj = json.load(open(os.path.join(args.model, "tokenizer.json"), encoding="utf-8"))
    model = tj["model"]
    vocab = model["vocab"]                         # token -> id
    merges = model.get("merges", [])
    added = tj.get("added_tokens", [])
    cfg = json.load(open(os.path.join(args.model, "config.json"), encoding="utf-8"))
    vocab_size = cfg.get("vocab_size", max(vocab.values()) + 1)
    print(f"base vocab {len(vocab)}, merges {len(merges)}, added {len(added)}, vocab_size {vocab_size}")

    n = max(vocab_size, max(vocab.values()) + 1, max((a["id"] for a in added), default=0) + 1)
    id2tok = [""] * n
    for tok, tid in vocab.items():
        id2tok[tid] = tok
    for a in added:                                # special tokens at their ids
        if 0 <= a["id"] < n:
            id2tok[a["id"]] = a["content"]
    for i in range(n):
        if id2tok[i] == "":
            id2tok[i] = f"<|unused_{i}|>"

    with open(os.path.join(args.output, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        for t in id2tok:
            # escape so line index == token id (some SentencePiece tokens contain literal \n/\r)
            t = t.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
            f.write(t + "\n")
    with open(os.path.join(args.output, "merges.txt"), "w", encoding="utf-8") as f:
        for mg in merges:
            f.write((" ".join(mg) if isinstance(mg, (list, tuple)) else mg) + "\n")
    json.dump({"added_tokens": added, "eos_token_id": 2, "pad_token_id": 0,
               "image_token_id": cfg.get("image_token_id"),
               "vision_start_token_id": cfg.get("vision_start_token_id"),
               "vision_end_token_id": cfg.get("vision_end_token_id")},
              open(os.path.join(args.output, "special_tokens.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"wrote vocab.txt ({n}), merges.txt ({len(merges)}), special_tokens.json -> {args.output}/")


if __name__ == "__main__":
    main()
