/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Adaptive Projected Guidance helpers (mirrors the top of
 * bernini/models/wan_diffusion.py: apg_delta, MomentumBuffer,
 * _normalize_diff, normalized_guidance, normalized_guidance_chain). */

#ifndef BERNINI_GUIDANCE_H
#define BERNINI_GUIDANCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double  momentum;
    float  *running; /* lazily allocated, starts at 0 */
    int64_t n;
} bn_momentum_t;

void bn_momentum_init(bn_momentum_t *mb, double momentum);
void bn_momentum_free(bn_momentum_t *mb);

/* apg_delta: project `delta` onto/off `ref` over the whole (batch-1) tensor
 * and recombine: parallel_scale * d_par + orthogonal_scale * d_orth.
 * In-place on delta. */
void bn_apg_delta(float *delta, const float *ref, int64_t n,
                  double parallel_scale, double orthogonal_scale);

/* _normalize_diff on spatial tensors [C,T,H,W] (batch 1): momentum update,
 * per-frame norm clamping, then per-frame parallel/orthogonal split against
 * base_pred with weight eta. In-place on diff. */
void bn_normalize_diff(float *diff, const float *base_pred, bn_momentum_t *mb,
                       double eta, double norm_threshold,
                       int c, int t, int h, int w);

/* normalized_guidance: out = pred_uncond + scale * nd(pred_cond - pred_uncond)
 * (out may alias none of the inputs). */
void bn_normalized_guidance(float *out, const float *pred_cond, const float *pred_uncond,
                            double guidance_scale, bn_momentum_t *mb, double eta,
                            double norm_threshold, int c, int t, int h, int w);

/* normalized_guidance_chain over `k` conditions: each condition's diff is
 * taken against the previous one. preds[k] are spatial tensors. */
void bn_normalized_guidance_chain(float *out, const float *pred_uncond,
                                  const float *const *preds, const double *scales,
                                  bn_momentum_t *mbs, double eta,
                                  const double *norm_thresholds, int k,
                                  int c, int t, int h, int w);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_GUIDANCE_H */
