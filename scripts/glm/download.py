#!/usr/bin/env python3
"""
Download GLM-OCR model from ModelScope (魔搭).

Usage:
    python download.py                    # download to default cache
    python download.py --output ./glm_ocr_model  # custom output dir
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download GLM-OCR from ModelScope")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: modelscope cache)")
    parser.add_argument("--model", type=str, default="ZhipuAI/glm-ocr",
                        help="ModelScope model ID")
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("ERROR: modelscope not installed. Run: pip install modelscope")
        sys.exit(1)

    print(f"Downloading {args.model} from ModelScope...")

    cache_dir = args.output
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    local_dir = snapshot_download(
        args.model,
        cache_dir=cache_dir,
        revision="master",
    )

    print(f"\nModel downloaded to: {local_dir}")

    # List top-level contents
    print("\nContents:")
    for item in sorted(os.listdir(local_dir)):
        full = os.path.join(local_dir, item)
        if os.path.isdir(full):
            print(f"  {item}/")
        else:
            size = os.path.getsize(full)
            if size > 1024 * 1024:
                print(f"  {item}  ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f"  {item}  ({size} B)")

    return local_dir


if __name__ == "__main__":
    main()