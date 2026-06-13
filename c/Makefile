# Bernini C inference port.
#
#   make            -> build ./bernini (CLI) and libbernini.a
#   make BLAS=1     -> link Apple Accelerate (macOS) for fast sgemm
#   make OMP=1      -> enable OpenMP (gcc, or clang with libomp)
#   make test       -> build + run unit tests

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

ifeq ($(BLAS),1)
  CFLAGS  += -DBT_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
  LDFLAGS += -framework Accelerate
endif

ifeq ($(OMP),1)
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
endif

SRC := src/bt.c src/bjson.c src/safetensors.c src/rng.c src/scheduler.c \
       src/attention.c src/transformer_wan.c src/guidance.c src/wan_diffusion.c \
       src/tokenizer.c src/umt5.c src/wan_vae.c src/data_utils.c src/media.c \
       src/pipeline.c
OBJ := $(SRC:.c=.o)

all: bernini libbernini.a

%.o: %.c
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

libbernini.a: $(OBJ)
	ar rcs $@ $^

bernini: $(OBJ) src/main.o
	$(CC) $^ -o $@ $(LDFLAGS)

test: tests/test_core
	./tests/test_core

tests/test_core: tests/test_core.c libbernini.a
	$(CC) $(CFLAGS) -Iinclude -Isrc $< libbernini.a -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ) src/main.o bernini libbernini.a tests/test_core

.PHONY: all test clean
