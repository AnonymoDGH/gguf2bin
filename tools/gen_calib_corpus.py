#!/usr/bin/env python3
"""Genera src/internal/calib_corpus.h desde tools/calib_corpus.txt.

Uso: python3 tools/gen_calib_corpus.py
El corpus calibra la criba del head (pack --fast-head). Los bytes no ASCII van
como escapes octales para no depender de la codificación de fuente del compilador.
"""
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = open(os.path.join(root, "tools", "calib_corpus.txt"), "rb").read()

def esc(b):
    out = []
    for c in b:
        if c == 0x5C: out.append("\\\\")
        elif c == 0x22: out.append('\\"')
        elif c == 0x0A: out.append("\\n")
        elif c == 0x09: out.append("\\t")
        elif c == 0x3F: out.append("\\?")          # evita trigrafos
        elif 0x20 <= c < 0x7F: out.append(chr(c))
        else: out.append("\\%03o" % c)
    return "".join(out)

lines = data.split(b"\n")
chunks, cur = [], b""
for i, ln in enumerate(lines):
    piece = ln + (b"\n" if i < len(lines) - 1 else b"")
    if len(cur) + len(piece) > 900 and cur:      # literales cortos (C99 garantiza 4095)
        chunks.append(cur); cur = b""
    cur += piece
if cur: chunks.append(cur)

with open(os.path.join(root, "src", "internal", "calib_corpus.h"), "w", newline="\n") as f:
    f.write("/* calib_corpus.h - GENERADO por tools/gen_calib_corpus.py desde\n"
            " * tools/calib_corpus.txt (%d bytes). No editar a mano.\n"
            " * Texto variado (prosa EN/ES, otros idiomas, codigo, chat, numeros) para\n"
            " * calibrar la criba del head. */\n" % len(data))
    f.write("#ifndef G2B_CALIB_CORPUS_H\n#define G2B_CALIB_CORPUS_H\n")
    f.write("static const char *const G2B_CALIB_CORPUS[] = {\n")
    for ch in chunks:
        f.write('  "%s",\n' % esc(ch))
    f.write("};\n#endif\n")
print("calib_corpus.h: %d bytes en %d literales" % (len(data), len(chunks)))
