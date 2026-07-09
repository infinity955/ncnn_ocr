"""Export HunyuanOCR tokenizer -> vocab.txt (120818 lines) + merges.txt for ncnn_llm.

Unlike export/extract_tokenizer.py (base vocab only), this also appends the 818 added
special tokens so that line index == token id for the full 120818-entry vocabulary that the
lm_head / embedding use.

Run under the conda `dev` env:
    python export/hunyuan_ocr_tokenizer.py [output_dir]
Default output_dir = assets/hunyuan_ocr.
"""
import sys, os, json, glob


def find_snapshot(model_dir=None):
    if model_dir and os.path.isdir(model_dir):
        return model_dir
    cands = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--tencent--HunyuanOCR/snapshots/*'))
    if not cands:
        raise SystemExit("HunyuanOCR HF snapshot not found in cache. Pass model_dir as second argument.")
    return cands[0]


def main():
    # model_dir: CLI arg > env var > relative default (../../../hunyuanocrmodel from this script = pnnx root)
    _model_default = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "hunyuanocrmodel"))
    if len(sys.argv) >= 3:
        model_dir = sys.argv[2]
    else:
        model_dir = os.environ.get("MODEL_PATH", _model_default)

    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'hunyuan_ocr')
    os.makedirs(out_dir, exist_ok=True)

    snap = find_snapshot(model_dir)
    tj = json.load(open(os.path.join(snap, 'tokenizer.json'), encoding='utf-8'))
    tc = json.load(open(os.path.join(snap, 'tokenizer_config.json'), encoding='utf-8'))

    model = tj.get('model', {})
    vocab = model.get('vocab', {})          # token -> id  (byte-level encoded)
    merges = model.get('merges', [])
    added = tc.get('added_tokens_decoder', {})  # id(str) -> {content, ...}

    # size = highest id + 1 across base vocab and added tokens
    max_id = max(vocab.values())
    for k in added.keys():
        max_id = max(max_id, int(k))
    size = max_id + 1

    id_to_token = [None] * size
    for tok, idx in vocab.items():
        id_to_token[idx] = tok
    for k, v in added.items():
        id_to_token[int(k)] = v['content']

    missing = [i for i, t in enumerate(id_to_token) if t is None]
    if missing:
        # keep line<->id alignment; fill gaps with a unique synthetic marker
        for i in missing:
            id_to_token[i] = f'<|unused_{i}|>'
        print(f'[warn] filled {len(missing)} gap id(s) with placeholders, e.g. {missing[:5]}')

    vocab_path = os.path.join(out_dir, 'vocab.txt')
    with open(vocab_path, 'w', encoding='utf-8', newline='\n') as f:
        for tok in id_to_token:
            f.write(tok + '\n')

    merges_path = os.path.join(out_dir, 'merges.txt')
    n_merges = 0
    with open(merges_path, 'w', encoding='utf-8', newline='\n') as f:
        for m in merges:
            if isinstance(m, list) and len(m) == 2:
                f.write(f'{m[0]} {m[1]}\n')
            elif isinstance(m, str):
                f.write(m + '\n')
            else:
                continue
            n_merges += 1

    print(f'[done] snapshot   : {snap}')
    print(f'[done] vocab.txt  : {vocab_path}  ({size} lines)')
    print(f'[done] merges.txt : {merges_path}  ({n_merges} pairs)')


if __name__ == '__main__':
    main()
