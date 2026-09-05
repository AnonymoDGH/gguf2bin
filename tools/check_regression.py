#!/usr/bin/env python3
"""check_regression.py — compara el bench actual contra un baseline JSON.

Uso:
  gguf2bin2 bench model.g2bx -n 32 > now.txt 2>&1
  python tools/check_regression.py now.txt baseline.json [--update report.json] [--tol 0.05]

Parsea líneas 'bench: N tokens in S s -> T tok/s' y 'bench-prefill: ...'.
Sale 0 si todo está dentro de ±tol (defecto 5%), 1 si hay regresión.
Con --update escribe el report (y crea el baseline si no existe).
"""
import json
import re
import sys

RE = re.compile(r"bench(?:-prefill)?:\s+(\d+) tokens in ([\d.]+)s?\s*->\s*([\d.]+) tok/s")


def parse(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE.search(line)
            if not m:
                continue
            kind = "prefill_tps" if "bench-prefill" in line else "decode_tps"
            out[kind] = float(m.group(3))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    now_path, base_path = sys.argv[1], sys.argv[2]
    update = None
    tol = 0.05
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--update" and i + 1 < len(args):
            update = args[i + 1]
            i += 2
        elif args[i] == "--tol" and i + 1 < len(args):
            tol = float(args[i + 1])
            i += 2
        else:
            i += 1
    now = parse(now_path)
    if not now:
        print("check_regression: no se pudo parsear", now_path)
        return 2
    try:
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        base = {}
    if not base and update:
        with open(update, "w", encoding="utf-8") as f:
            json.dump(now, f, indent=1)
        print("check_regression: baseline creado", now)
        return 0
    bad = 0
    for k, v in now.items():
        b = base.get(k)
        if not b:
            print(f"check_regression: sin baseline para {k} (ahora {v})")
            continue
        delta = (v - b) / b
        flag = "OK " if delta >= -tol else "REGRESION"
        if delta < -tol:
            bad += 1
        print(f"check_regression: {k}: base={b:.1f} ahora={v:.1f} delta={delta*100:+.1f}% [{flag}]")
    if update:
        with open(update, "w", encoding="utf-8") as f:
            json.dump(now, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
