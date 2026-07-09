"""Add KV-cache support to the PaddleOCR-VL ncnn decoder param.

The exported decoder runs the full sequence every step (SDPA has 4 inputs / 1 output).
ncnn's SDPA layer natively supports incremental KV cache: 6 inputs
(q, k, v, mask, cache_k, cache_v) -> 3 outputs (out, out_cache_k, out_cache_v), with params
5=has_mask, 6=scale, 7=kvcache_enabled -- exactly what src/glm_ocr.cpp / text_runtime.cpp expect.

Rewrites every SDPA op to that signature, adds a single `Input kv_cache` layer producing
cache_k{i}/cache_v{i} for all layers, and fixes the header layer/blob counts.

Usage: .venv-glm/Scripts/python.exe scripts/glm/add_kvcache.py [path_to_pdvl_decoder.ncnn.param]
A backup `<param>.nokv` is written the first time.

(Ported from scripts/hunyuan/add_kvcache.py -- the working, format-correct version.)
"""
import sys, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
DEF = os.path.join(HERE, "ncnn", "pdvl_decoder.ncnn.param")
HEAD_DIM = 128
SCALE = "%g" % (1.0 / math.sqrt(HEAD_DIM))  # 0.0883883


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEF
    raw = open(path, "r", encoding="utf-8").read()
    lines = raw.split("\n")
    if lines[0].strip() != "7767517":
        raise SystemExit("not an ncnn param file: " + path)

    bak = path + ".nokv"
    if not os.path.exists(bak):
        open(bak, "w", encoding="utf-8").write(raw)
        print("[backup]", bak)

    header = lines[1].split()
    orig_layers, orig_blobs = int(header[0]), int(header[1])

    body = lines[2:]

    def noutputs(l):
        f = l.split()
        return int(f[3]) if len(f) >= 4 else 0

    layer_lines = [l for l in body if l.strip()]
    calc_blobs = sum(noutputs(l) for l in layer_lines)
    print(f"[check] header layers={orig_layers} blobs={orig_blobs}; computed blobs={calc_blobs}")

    in3_idx = None
    for i, l in enumerate(body):
        f = l.split()
        if len(f) >= 5 and f[0] == "Input" and f[-1] == "in3":
            in3_idx = i
            break
    if in3_idx is None:
        raise SystemExit("could not find `Input in3` line")

    sdpa_i = 0
    out_body = []
    for l in body:
        f = l.split()
        if f and f[0] == "SDPA":
            name = f[1]
            nin, nout = int(f[2]), int(f[3])
            blobs = f[4:4 + nin + nout]
            inputs = blobs[:nin]        # q, k, v, mask
            outputs = blobs[nin:]       # out
            params = f[4 + nin + nout:]
            pd = {}
            for p in params:
                k, v = p.split("=")
                pd[k] = v
            pd["5"] = "1"
            pd["6"] = SCALE
            pd["7"] = "1"
            new_in = inputs + [f"cache_k{sdpa_i}", f"cache_v{sdpa_i}"]
            new_out = outputs + [f"out_cache_k{sdpa_i}", f"out_cache_v{sdpa_i}"]
            pstr = " ".join(f"{k}={pd[k]}" for k in sorted(pd, key=lambda x: int(x)))
            nl = "%-24s %-24s %d %d %s %s" % (
                "SDPA", name, len(new_in), len(new_out),
                " ".join(new_in + new_out), pstr)
            out_body.append(nl)
            sdpa_i += 1
        else:
            out_body.append(l)

    num_layers = sdpa_i
    cache_blobs = []
    for i in range(num_layers):
        cache_blobs += [f"cache_k{i}", f"cache_v{i}"]
    input_line = "%-24s %-24s 0 %d %s" % ("Input", "kv_cache", len(cache_blobs),
                                          " ".join(cache_blobs))
    out_body.insert(in3_idx + 1, input_line)

    new_layer_lines = [l for l in out_body if l.strip()]
    new_layers = len(new_layer_lines)
    new_blobs = sum(noutputs(l) for l in new_layer_lines)

    new_lines = [lines[0], "%d %d" % (new_layers, new_blobs)] + out_body
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(new_lines))
    print(f"[done] SDPA rewritten: {num_layers}; header -> {new_layers} {new_blobs}")
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
