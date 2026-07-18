# gguf2bin2 — G2BX runtime (C99) — FIXED v3.3
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -std=c99 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm -fopenmp

SRC = src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c \
      src/l5_model.c src/l6_token.c src/main.c
BIN = gguf2bin2
ifeq ($(OS),Windows_NT)
  EXE = $(BIN).exe
  TEST_MODEL = tiny_test.g2bx
else
  EXE = $(BIN)
  TEST_MODEL = /tmp/tiny_test.g2bx
endif

.PHONY: all clean tiny test rebuild

all: $(EXE)

$(EXE): $(SRC) include/g2b.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

tiny: CFLAGS=-Os -std=c99 -ffunction-sections -fdata-sections -Iinclude -Wall
tiny: LDFLAGS=-lm -Wl,--gc-sections -s
tiny: $(EXE)

# Portable smoke test (Windows + Unix)
test: $(EXE)
	./$(EXE) synth $(TEST_MODEL)
	./$(EXE) info $(TEST_MODEL)
	./$(EXE) run $(TEST_MODEL) -n 4 -t 0
	./$(EXE) bench $(TEST_MODEL) -n 8
	@echo "test OK"

rebuild: clean all

clean:
	rm -f $(BIN) $(BIN).exe *.o src/*.o tiny_test.g2bx _check_tiny.g2bx
	-rm -f /tmp/tiny_test.g2bx 2>/dev/null || true

# Windows (MinGW) sin OpenMP / native si falla:
#   gcc -O2 -std=c99 -Iinclude -o gguf2bin2.exe src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c src/l5_model.c src/l6_token.c src/main.c -lm
