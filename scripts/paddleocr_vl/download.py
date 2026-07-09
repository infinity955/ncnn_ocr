#!/usr/bin/env python3
"""
Download PaddleOCR-VL-1.6 from ModelScope (魔搭).

PaddleOCR-VL-1.6 (2026-05-28): 0.9B VLM document parser, OmniDocBench v1.6 96.33% SOTA.
Architecture: NaViT vision encoder (~600M) + 2-layer MLP projector + ERNIE-4.5-0.3B decoder.

Usage:
    .venv-glm/Scripts/python.exe scripts/paddleocr_vl/download.py --output ../paddleocr_vl_model
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Download PaddleOCR-VL-1.6 from ModelScope")
    ap.add_argument("--output", "-o", default=None, help="Output dir (default: modelscope cache)")
    ap.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6", help="ModelScope model ID")
    args = ap.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("ERROR: modelscope not installed. Run: pip install modelscope")
        sys.exit(1)

    print(f"Downloading {args.model} from ModelScope ...")
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    local_dir = snapshot_download(args.model, cache_dir=args.output, revision="master")
    print(f"\nModel downloaded to: {local_dir}\n\nContents:")
    for item in sorted(os.listdir(local_dir)):
        full = os.path.join(local_dir, item)
        if os.path.isdir(full):
            print(f"  {item}/")
        else:
            sz = os.path.getsize(full)
            print(f"  {item}  ({sz/1024/1024:.1f} MB)" if sz > 1024*1024 else f"  {item}  ({sz} B)")


if __name__ == "__main__":
    main()
