/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* UMT5 encoder (replaces transformers.UMT5EncoderModel for the Wan text
 * encoder). UMT5 differs from T5 in that every layer carries its own
 * relative attention bias. */

#ifndef BERNINI_UMT5_H
#define BERNINI_UMT5_H

#include "bt.h"
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bt_t *ln1_w;                 /* layer.0.layer_norm */
    bt_t *q_w, *k_w, *v_w, *o_w; /* SelfAttention (no biases) */
    bt_t *rel_bias;              /* relative_attention_bias [buckets, heads] */
    bt_t *ln2_w;                 /* layer.1.layer_norm */
    bt_t *wi0_w, *wi1_w, *wo_w;  /* DenseGatedActDense */
} umt5_block_t;

typedef struct {
    int d_model, d_kv, d_ff, num_layers, num_heads;
    int rel_buckets, rel_max_distance;
    float eps;
    bt_t *shared;        /* token embeddings [vocab, d_model] */
    umt5_block_t *blocks;
    bt_t *final_ln_w;
} umt5_t;

/* Load from a directory holding config.json + *.safetensors. */
int  umt5_load(umt5_t *m, const char *dir);
void umt5_free(umt5_t *m);

/* Encoder forward for one sequence: ids/mask [len]; returns the last hidden
 * state [len, d_model] (caller frees). */
float *umt5_encode(const umt5_t *m, const int32_t *ids, const int32_t *mask, int len);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_UMT5_H */
