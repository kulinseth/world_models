/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "attention.h"
#include "bt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* One (segment, head): out[lq, hd] = softmax(q k^T * scale) v
 * q strided by [H*D] rows (packed layout [tokens, H, D]). */
static void attn_one_head(const float *q, const float *k, const float *v, float *out,
                          int64_t lq, int64_t lk, int stride, int hd, float scale,
                          int causal, const float *bias_qk, const float *key_mask_neg) {
    float *scores = (float *)malloc(sizeof(float) * (size_t)lk);
    for (int64_t i = 0; i < lq; i++) {
        const float *qi = q + i * stride;
        float mx = -INFINITY;
        int64_t kmax = causal ? (lk - lq + i + 1) : lk;
        if (kmax > lk) kmax = lk;
        for (int64_t j = 0; j < kmax; j++) {
            const float *kj = k + j * stride;
            double dot = 0.0;
            for (int d = 0; d < hd; d++) dot += (double)qi[d] * (double)kj[d];
            float sc = (float)(dot * scale);
            if (bias_qk) sc += bias_qk[i * lk + j];
            if (key_mask_neg) sc += key_mask_neg[j];
            scores[j] = sc;
            if (sc > mx) mx = sc;
        }
        double sum = 0.0;
        for (int64_t j = 0; j < kmax; j++) {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        float inv = (float)(1.0 / sum);
        float *oi = out + i * stride;
        for (int d = 0; d < hd; d++) oi[d] = 0.0f;
        for (int64_t j = 0; j < kmax; j++) {
            float w = scores[j] * inv;
            if (w == 0.0f) continue;
            const float *vj = v + j * stride;
            for (int d = 0; d < hd; d++) oi[d] += w * vj[d];
        }
    }
    free(scores);
}

void bn_varlen_attention(const float *q, const float *k, const float *v, float *out,
                         const int64_t *cu_q, const int64_t *cu_k,
                         int nseg, int heads, int hd, int causal) {
    const int stride = heads * hd;
    const float scale = 1.0f / sqrtf((float)hd);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int s = 0; s < nseg; s++) {
        for (int h = 0; h < heads; h++) {
            int64_t q0 = cu_q[s], q1 = cu_q[s + 1];
            int64_t k0 = cu_k[s], k1 = cu_k[s + 1];
            attn_one_head(q + q0 * stride + (int64_t)h * hd,
                          k + k0 * stride + (int64_t)h * hd,
                          v + k0 * stride + (int64_t)h * hd,
                          out + q0 * stride + (int64_t)h * hd,
                          q1 - q0, k1 - k0, stride, hd, scale, causal, NULL, NULL);
        }
    }
}

void bn_attention_bias(const float *q, const float *k, const float *v, float *out,
                       int lq, int lk, int heads, int hd,
                       const float *bias, const float *key_mask_neg, float scale) {
    const int stride = heads * hd;
#pragma omp parallel for schedule(dynamic)
    for (int h = 0; h < heads; h++) {
        attn_one_head(q + (int64_t)h * hd, k + (int64_t)h * hd, v + (int64_t)h * hd,
                      out + (int64_t)h * hd, lq, lk, stride, hd, scale, 0,
                      bias ? bias + (int64_t)h * lq * lk : NULL, key_mask_neg);
    }
}
