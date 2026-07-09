#!/usr/bin/env python3
"""
Inspect GLM-OCR model architecture.

Loads the model from ModelScope cache (or a local path) and prints:
- Full module tree
- Parameter shapes
- Forward signatures
- Key architecture details needed for ncnn export

Usage:
    python inspect_model.py                          # auto-detect from modelscope cache
    python inspect_model.py --model ./glm_ocr_model  # explicit path
"""

import argparse
import os
import sys

import torch
import numpy as np


def find_model_path():
    """Auto-detect model path (repo-local download or modelscope cache)."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))            # scripts/glm
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))  # pnnx root
    # repo-local download
    candidate = os.path.join(repo, "glm_ocr_model", "ZhipuAI", "glm-ocr")
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
        return candidate
    # modelscope cache
    home = os.path.expanduser("~")
    cache_base = os.path.join(home, ".cache", "modelscope", "hub", "models", "ZhipuAI", "glm-ocr")
    if os.path.isdir(cache_base):
        return cache_base
    return None


def print_module_tree(module, prefix="", max_depth=5, current_depth=0):
    """Print the module hierarchy tree."""
    if current_depth >= max_depth:
        return

    for name, child in module.named_children():
        # Count parameters
        total_params = sum(p.numel() for p in child.parameters())
        total_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)

        info = f"{type(child).__name__}"
        if total_params > 0:
            if total_params >= 1e9:
                info += f"  [{total_params/1e9:.2f}B params]"
            elif total_params >= 1e6:
                info += f"  [{total_params/1e6:.1f}M params]"
            else:
                info += f"  [{total_params} params]"

        print(f"{prefix}├── {name}: {info}")

        # Check if this child has further children
        grandchildren = list(child.named_children())
        if grandchildren:
            new_prefix = prefix + "│   "
            print_module_tree(child, new_prefix, max_depth, current_depth + 1)


def print_param_shapes(module, prefix=""):
    """Print all parameter shapes."""
    for name, param in module.named_parameters():
        shape_str = " × ".join(str(d) for d in param.shape)
        print(f"  {prefix}{name}: [{shape_str}]  ({param.dtype})")


def print_buffer_shapes(module, prefix=""):
    """Print all buffer shapes."""
    for name, buf in module.named_buffers():
        shape_str = " × ".join(str(d) for d in buf.shape)
        print(f"  {prefix}[buffer] {name}: [{shape_str}]  ({buf.dtype})")


def inspect_model(model):
    """Print comprehensive model architecture info."""
    print("=" * 70)
    print("MODEL CLASS:", type(model).__name__)
    print("=" * 70)

    # Print config
    print("\n--- CONFIG ---")
    if hasattr(model, 'config'):
        config = model.config
        for key in sorted(config.to_dict().keys()):
            val = getattr(config, key, None)
            if isinstance(val, (list, tuple)) and len(str(val)) > 80:
                print(f"  {key}: [...] (len={len(val)})")
            else:
                print(f"  {key}: {val}")

    # Print module tree
    print("\n--- MODULE TREE (top-level) ---")
    print_module_tree(model, max_depth=1)

    # Find key submodules
    print("\n--- KEY SUBMODULES ---")

    # Vision encoder
    vision_candidates = []
    for name, mod in model.named_children():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ['vision', 'vit', 'visual', 'image']):
            vision_candidates.append(name)

    if vision_candidates:
        print(f"\nVision encoder candidates: {vision_candidates}")
        for vc in vision_candidates:
            mod = getattr(model, vc)
            print(f"\n  [{vc}] subtree:")
            print_module_tree(mod, prefix="  ", max_depth=3, current_depth=0)
            print(f"\n  [{vc}] parameters:")
            print_param_shapes(mod, prefix="  ")
    else:
        print("\nNo obvious vision encoder found. Full module tree:")
        print_module_tree(model, max_depth=3)

    # Check for transformer/LLM backbone
    llm_candidates = []
    for name, mod in model.named_children():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ['transformer', 'model', 'llm', 'decoder', 'language']):
            llm_candidates.append(name)

    if llm_candidates:
        print(f"\nLLM backbone candidates: {llm_candidates}")
        for lc in llm_candidates[:1]:  # Just first one
            mod = getattr(model, lc)
            print(f"\n  [{lc}] subtree:")
            print_module_tree(mod, prefix="  ", max_depth=2, current_depth=0)

            # Check for attention layers
            for name2, child in mod.named_modules():
                if 'attention' in name2.lower() or 'self_attn' in name2.lower() or 'attn' in name2.lower():
                    print(f"\n  [{lc}.{name2}] parameters:")
                    print_param_shapes(child, prefix="    ")
                    # Print first attention layer only
                    break

    # Check for embed_tokens
    for name, mod in model.named_modules():
        if name.endswith('embed_tokens') or name.endswith('word_embedding') or name.endswith('wte'):
            print(f"\n  Embedding layer: {name}")
            print_param_shapes(mod, prefix="    ")
            break

    # Check for lm_head
    for name, mod in model.named_modules():
        if name == 'lm_head' or name.endswith('.lm_head'):
            print(f"\n  LM Head: {name}")
            print_param_shapes(mod, prefix="    ")
            break

    # Check for final norm
    for name, mod in model.named_modules():
        if name.endswith('final_layernorm') or name.endswith('norm') or name.endswith('ln_f'):
            if hasattr(mod, 'weight') and mod.weight is not None:
                print(f"\n  Final norm: {name}")
                print_param_shapes(mod, prefix="    ")
                break

    # Print total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- TOTAL PARAMETERS ---")
    print(f"  Total: {total/1e9:.2f}B ({total:,})")
    print(f"  Trainable: {trainable/1e9:.2f}B ({trainable:,})")

    # Try a forward pass to check shapes
    print("\n--- FORWARD PASS TEST ---")
    try:
        # Try with dummy text input
        dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=dummy_input_ids)
        if hasattr(output, 'logits'):
            print(f"  Text-only forward: logits shape = {list(output.logits.shape)}")
        elif hasattr(output, 'last_hidden_state'):
            print(f"  Text-only forward: last_hidden_state shape = {list(output.last_hidden_state.shape)}")
        else:
            print(f"  Text-only forward: output type = {type(output).__name__}")
            if isinstance(output, torch.Tensor):
                print(f"    shape = {list(output.shape)}")
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        print(f"    [{i}] shape = {list(o.shape)}")
    except Exception as e:
        print(f"  Text-only forward failed: {e}")

    # Try with dummy pixel values
    try:
        dummy_pixels = torch.randn(1, 3, 448, 448)
        dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=dummy_input_ids, pixel_values=dummy_pixels)
        if hasattr(output, 'logits'):
            print(f"  Vision+text forward: logits shape = {list(output.logits.shape)}")
        elif hasattr(output, 'last_hidden_state'):
            print(f"  Vision+text forward: last_hidden_state shape = {list(output.last_hidden_state.shape)}")
        else:
            print(f"  Vision+text forward: output type = {type(output).__name__}")
    except Exception as e:
        print(f"  Vision+text forward failed: {e}")

    # Try processor
    print("\n--- PROCESSOR / TOKENIZER ---")
    try:
        from transformers import AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)
        print(f"  Processor type: {type(processor).__name__}")
        if hasattr(processor, 'tokenizer'):
            tok = processor.tokenizer
            print(f"  Tokenizer vocab size: {tok.vocab_size}")
            print(f"  Tokenizer type: {type(tok).__name__}")
            # Check special tokens
            for attr in ['bos_token', 'eos_token', 'pad_token', 'unk_token',
                         'image_token', 'image_start_token', 'image_end_token']:
                if hasattr(tok, attr):
                    val = getattr(tok, attr)
                    if val is not None:
                        tid = tok.convert_tokens_to_ids(val) if isinstance(val, str) else '?'
                        print(f"  {attr}: '{val}' -> id={tid}")
    except Exception as e:
        print(f"  Failed to load processor: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect GLM-OCR model architecture")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model directory (auto-detect from modelscope cache if not set)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True,
                        help="Trust remote code (default: True)")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = find_model_path()
        if model_path is None:
            print("ERROR: Could not auto-detect model path.")
            print("Please specify with --model or download first with download.py")
            sys.exit(1)

    print(f"Loading model from: {model_path}")

    from transformers import AutoModel, AutoModelForCausalLM
    import json

    # AutoModelForVision2Seq was renamed to AutoModelForImageTextToText (transformers >=5.0)
    try:
        from transformers import AutoModelForImageTextToText as AutoModelVL
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelVL

    # Load config first
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        arch = config.get("architectures", ["unknown"])[0]
        print(f"Architecture: {arch}")

    # Try different model classes
    model = None
    for model_cls in [AutoModelVL, AutoModelForCausalLM, AutoModel]:
        try:
            print(f"Trying {model_cls.__name__}...")
            model = model_cls.from_pretrained(
                model_path,
                trust_remote_code=args.trust_remote_code,
                dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            print(f"Loaded with {model_cls.__name__}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if model is None:
        print("ERROR: Could not load model with any known class.")
        sys.exit(1)

    model.eval()
    inspect_model(model)


if __name__ == "__main__":
    main()