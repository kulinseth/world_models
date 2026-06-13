/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Variable-length attention (mirrors bernini/attention.py).
 * q/k/v are packed as [total_tokens, num_heads, head_dim] and cu_seqlens give
 * per-sample offsets into total_tokens. The C build always uses the SDPA-style
 * reference path (the FlashAttention backends are CUDA-only). */

#ifndef BERNINI_ATTENTION_H
#define BERNINI_ATTENTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* out: [total_q_tokens, num_heads, head_dim]. Softmax(QK^T/sqrt(d))V per
 * (segment, head). `causal` supports the causal masking used by LLM-style
 * components. */
void bn_varlen_attention(const float *q, const float *k, const float *v, float *out,
                         const int64_t *cu_seqlens_q, const int64_t *cu_seqlens_k,
                         int num_segments, int num_heads, int head_dim, int causal);

/* Dense single-segment attention with an optional additive bias
 * [num_heads, lq, lk] (used by the UMT5 encoder's relative position bias).
 * Layout: q [lq, H, D], k/v [lk, H, D], out [lq, H, D].
 * `scale` multiplies the logits (T5 uses 1.0, SDPA uses 1/sqrt(d)). */
void bn_attention_bias(const float *q, const float *k, const float *v, float *out,
                       int lq, int lk, int num_heads, int head_dim,
                       const float *bias, const float *key_mask_neg, float scale);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_ATTENTION_H */
