/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "rng.h"

#include <math.h>

void bn_rng_seed(bn_rng *r, uint64_t seed) {
    r->mt[0] = (uint32_t)(seed & 0xFFFFFFFFu);
    for (int i = 1; i < 624; i++)
        r->mt[i] = (uint32_t)(1812433253u * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i);
    r->idx = 624;
    r->has_spare = 0;
    r->spare = 0.0;
}

static uint32_t mt_next(bn_rng *r) {
    if (r->idx >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (r->mt[i] & 0x80000000u) | (r->mt[(i + 1) % 624] & 0x7FFFFFFFu);
            r->mt[i] = r->mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1) r->mt[i] ^= 2567483615u;
        }
        r->idx = 0;
    }
    uint32_t y = r->mt[r->idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 2636928640u;
    y ^= (y << 15) & 4022730752u;
    y ^= y >> 18;
    return y;
}

double bn_rng_uniform(bn_rng *r) {
    /* 53-bit resolution in [0,1) */
    uint32_t a = mt_next(r) >> 5, b = mt_next(r) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

double bn_rng_normal(bn_rng *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    double u, v, s;
    do {
        u = 2.0 * bn_rng_uniform(r) - 1.0;
        v = 2.0 * bn_rng_uniform(r) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double m = sqrt(-2.0 * log(s) / s);
    r->spare = v * m;
    r->has_spare = 1;
    return u * m;
}

void bn_rng_randn(bn_rng *r, float *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (float)bn_rng_normal(r);
}

void bn_rng_shuffle(bn_rng *r, int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(bn_rng_uniform(r) * (i + 1));
        if (j > i) j = i;
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}
