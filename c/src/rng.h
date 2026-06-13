/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* rng: seeded MT19937 with Gaussian sampling.
 * Replaces torch.Generator(device="cpu").manual_seed(seed) + randn_tensor.
 * Note: the stream is not bit-identical to torch's, so a given seed yields a
 * different (but equally valid) noise sample than the Python pipeline. */

#ifndef BERNINI_RNG_H
#define BERNINI_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t mt[624];
    int      idx;
    int      has_spare;
    double   spare;
} bn_rng;

void   bn_rng_seed(bn_rng *r, uint64_t seed);
double bn_rng_uniform(bn_rng *r);              /* [0,1) */
double bn_rng_normal(bn_rng *r);               /* N(0,1), Box-Muller */
void   bn_rng_randn(bn_rng *r, float *out, int64_t n);
void   bn_rng_shuffle(bn_rng *r, int *idx, int n); /* Fisher-Yates */

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_RNG_H */
