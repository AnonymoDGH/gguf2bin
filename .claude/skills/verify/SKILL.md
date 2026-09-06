# CLI Verification

Build on Windows with `mingw32-make`. Run commands sequentially when they
create or consume the same binary or model. Do not run `clean` while testing.

- `gguf2bin2.exe synth tiny_test.g2bx` creates a disposable model.
- `gguf2bin2.exe run tiny_test.g2bx -n 16 -t 0.7 --seed 42 --top-k 40 --top-p 0.9`
  must print `1 92 94 71 80 75 88 79 76 112 70 66 86 76 75 116 76`.
- `gguf2bin2.exe verify tiny_test.g2bx` checks the serialized format.
- `gguf2bin2.exe run tiny_test.g2bx --mv 0.5` and the equivalent `--bvh`
  command must reject the removed options with exit 1, not silently ignore them.
- `gguf2bin2.exe cyber-train` must reject the removed command.

Use UTF-8 text for ppl. PowerShell 5.1 redirection writes UTF-16 by default.
The tiny model has no tokenizer and head_dim=16: it cannot validate text ppl
or actual Q8 KV arithmetic. Use a real model with compatible geometry for those.
