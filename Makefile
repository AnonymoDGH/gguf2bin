# gguf2bin2 — G2BX runtime (C99) — v4.7
CC      ?= gcc
CFLAGS  ?= -O3 -mavx2 -mfma -mf16c -ffast-math -fopenmp -std=c99 -Wall -Wextra -Iinclude -IVulkan-Headers-1.3.290/include
LDFLAGS ?= -lm -fopenmp

CORE = src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c \
       src/l5_model.c src/l6_token.c src/l7_vulkan.c src/l8_cyber.c
SRC = $(CORE) src/main.c
BIN = gguf2bin2
ifeq ($(OS),Windows_NT)
  EXE = $(BIN).exe
  KVEXE = tools/kvtest.exe
  PFEXE = tools/prefilltest.exe
  QBEXE = tools/q4bcheck.exe
  IQEXE = tools/iq1check.exe
  QKEXE = tools/q3kcheck.exe
  TEST_MODEL = tiny_test.g2bx
else
  EXE = $(BIN)
  KVEXE = tools/kvtest
  PFEXE = tools/prefilltest
  QBEXE = tools/q4bcheck
  IQEXE = tools/iq1check
  QKEXE = tools/q3kcheck
  TEST_MODEL = /tmp/tiny_test.g2bx
  LDFLAGS += -ldl
endif

# Herramienta: verificacion numerica KV cache F32 vs Q8_0 (misma arquitectura)
KVSRC = tools/kvtest.c $(CORE)

.PHONY: all clean tiny test kvtest prefilltest q4bcheck iq1check q3kcheck rebuild

all: $(EXE)

$(EXE): $(SRC) include/g2b.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)
ifeq ($(OS),Windows_NT)
	copy /Y $@ gguf2bin.exe
endif

tiny: tiny.exe
tiny.exe: $(SRC) include/g2b.h
	$(CC) -Os -std=c99 -ffunction-sections -fdata-sections -Iinclude -Wall -o $@ $(SRC) -lm -Wl,--gc-sections -s

# Portable smoke test (Windows + Unix)
test: $(EXE)
	./$(EXE) synth $(TEST_MODEL)
	./$(EXE) info $(TEST_MODEL)
	./$(EXE) run $(TEST_MODEL) -n 4 -t 0
	./$(EXE) bench $(TEST_MODEL) -n 8
	@echo "test OK"

# Valida equivalencia F32 vs Q8 KV sobre un modelo (por defecto el sintetico)
kvtest: $(KVEXE)
	./$(KVEXE) $(TEST_MODEL) 64

$(KVEXE): $(KVSRC)
	$(CC) $(CFLAGS) -o $@ $(KVSRC) $(LDFLAGS)

PFSRC = tools/prefilltest.c $(CORE)
prefilltest: $(PFEXE)
$(PFEXE): $(PFSRC) include/g2b.h
	$(CC) $(CFLAGS) -o $@ $(PFSRC) $(LDFLAGS)

QBSRC = tools/q4bcheck.c $(CORE)
q4bcheck: $(QBEXE)
	./$(QBEXE)

IQSRC = tools/iq1check.c $(CORE)
iq1check: $(IQEXE)
	./$(IQEXE)
$(IQEXE): $(IQSRC) include/g2b.h
	$(CC) $(CFLAGS) -o $@ $(IQSRC) $(LDFLAGS)

QKSRC = tools/q3kcheck.c $(CORE)
q3kcheck: $(QKEXE)
	./$(QKEXE)
$(QKEXE): $(QKSRC) include/g2b.h
	$(CC) $(CFLAGS) -o $@ $(QKSRC) $(LDFLAGS)
$(QBEXE): $(QBSRC) include/g2b.h
	$(CC) $(CFLAGS) -o $@ $(QBSRC) $(LDFLAGS)

rebuild: clean all

ifeq ($(OS),Windows_NT)
clean:
	-del /Q $(BIN).exe tiny.exe gguf2bin.exe tiny_test.g2bx _check_tiny.g2bx 2>NUL
	-del /Q src\*.o 2>NUL
	-del /Q tools\kvtest.exe tools\prefilltest.exe tools\q4bcheck.exe tools\iq1check.exe tools\q3kcheck.exe 2>NUL
#else
clean:
	rm -f $(BIN) $(BIN).exe tiny.exe tiny *.o src/*.o tiny_test.g2bx _check_tiny.g2bx $(KVEXE) $(PFEXE) $(QBEXE) $(IQEXE) $(QKEXE)
	-rm -f /tmp/tiny_test.g2bx 2>/dev/null || true
endif

# Windows (MinGW) sin OpenMP / native si falla:
#   gcc -O2 -std=c99 -Iinclude -o gguf2bin2.exe $(CORE) src/main.c -lm
