/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "guidance.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void bn_momentum_init(bn_momentum_t *mb, double momentum) {
    mb->momentum = momentum;
    mb->running = NULL;
    mb->n = 0;
}

void bn_momentum_free(bn_momentum_t *mb) {
    free(mb->running);
    mb->running = NULL;
    mb->n = 0;
}

static void momentum_update(bn_momentum_t *mb, float *diff, int64_t n) {
    if (!mb->running) {
        mb->running = (float *)calloc((size_t)n, sizeof(float));
        mb->n = n;
    }
    /* running = diff + momentum * running; diff <- running */
    for (int64_t i = 0; i < n; i++) {
        mb->running[i] = diff[i] + (float)mb->momentum * mb->running[i];
        diff[i] = mb->running[i];
    }
}

void bn_apg_delta(float *delta, const float *ref, int64_t n,
                  double parallel_scale, double orthogonal_scale) {
    double ref_sq = 0.0, dot = 0.0;
    for (int64_t i = 0; i < n; i++) {
        ref_sq += (double)ref[i] * (double)ref[i];
        dot += (double)delta[i] * (double)ref[i];
    }
    if (ref_sq < 1e-8) ref_sq = 1e-8;
    double coeff = dot / ref_sq;
    for (int64_t i = 0; i < n; i++) {
        double par = coeff * (double)ref[i];
        double orth = (double)delta[i] - par;
        delta[i] = (float)(parallel_scale * par + orthogonal_scale * orth);
    }
}

/* dims [-1,-2,-4] of [B=1,C,T,H,W] => reduce over (C,H,W), keep T */
void bn_normalize_diff(float *diff, const float *base, bn_momentum_t *mb,
                       double eta, double norm_threshold,
                       int c, int t, int h, int w) {
    const int64_t hw = (int64_t)h * w;
    const int64_t thw = (int64_t)t * hw;
    const int64_t n = (int64_t)c * thw;

    if (mb) momentum_update(mb, diff, n);

    if (norm_threshold > 0) {
        for (int ti = 0; ti < t; ti++) {
            double ss = 0.0;
            for (int ci = 0; ci < c; ci++) {
                const float *p = diff + (int64_t)ci * thw + (int64_t)ti * hw;
                for (int64_t i = 0; i < hw; i++) ss += (double)p[i] * (double)p[i];
            }
            double nrm = sqrt(ss);
            double sf = nrm > 0 ? norm_threshold / nrm : 1.0;
            if (sf > 1.0) sf = 1.0;
            for (int ci = 0; ci < c; ci++) {
                float *p = diff + (int64_t)ci * thw + (int64_t)ti * hw;
                for (int64_t i = 0; i < hw; i++) p[i] = (float)((double)p[i] * sf);
            }
        }
    }

    /* per-frame projection of diff onto the normalized base prediction */
    for (int ti = 0; ti < t; ti++) {
        double bss = 0.0, dot = 0.0;
        for (int ci = 0; ci < c; ci++) {
            const float *bp = base + (int64_t)ci * thw + (int64_t)ti * hw;
            const float *dp = diff + (int64_t)ci * thw + (int64_t)ti * hw;
            for (int64_t i = 0; i < hw; i++) {
                bss += (double)bp[i] * (double)bp[i];
                dot += (double)dp[i] * (double)bp[i];
            }
        }
        double bn = sqrt(bss);
        if (bn < 1e-12) bn = 1e-12;
        /* v1 = base/||base||; parallel = (d.v1) v1; out = orth + eta*parallel */
        double proj = dot / (bn * bn);
        for (int ci = 0; ci < c; ci++) {
            const float *bp = base + (int64_t)ci * thw + (int64_t)ti * hw;
            float *dp = diff + (int64_t)ci * thw + (int64_t)ti * hw;
            for (int64_t i = 0; i < hw; i++) {
                double par = proj * (double)bp[i];
                double orth = (double)dp[i] - par;
                dp[i] = (float)(orth + eta * par);
            }
        }
    }
}

void bn_normalized_guidance(float *out, const float *pred_cond, const float *pred_uncond,
                            double scale, bn_momentum_t *mb, double eta,
                            double norm_threshold, int c, int t, int h, int w) {
    const int64_t n = (int64_t)c * t * h * w;
    float *diff = (float *)malloc(sizeof(float) * (size_t)n);
    for (int64_t i = 0; i < n; i++) diff[i] = pred_cond[i] - pred_uncond[i];
    bn_normalize_diff(diff, pred_cond, mb, eta, norm_threshold, c, t, h, w);
    for (int64_t i = 0; i < n; i++) out[i] = pred_uncond[i] + (float)scale * diff[i];
    free(diff);
}

void bn_normalized_guidance_chain(float *out, const float *pred_uncond,
                                  const float *const *preds, const double *scales,
                                  bn_momentum_t *mbs, double eta,
                                  const double *norm_thresholds, int k,
                                  int c, int t, int h, int w) {
    const int64_t n = (int64_t)c * t * h * w;
    float *diff = (float *)malloc(sizeof(float) * (size_t)n);
    memcpy(out, pred_uncond, sizeof(float) * (size_t)n);
    for (int i = 0; i < k; i++) {
        const float *base = (i == 0) ? pred_uncond : preds[i - 1];
        for (int64_t j = 0; j < n; j++) diff[j] = preds[i][j] - base[j];
        bn_normalize_diff(diff, preds[i], &mbs[i], eta, norm_thresholds[i], c, t, h, w);
        for (int64_t j = 0; j < n; j++) out[j] += (float)scales[i] * diff[j];
    }
    free(diff);
}
