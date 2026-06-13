/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* WanTransformer3DModel (mirrors bernini/models/transformer_wan.py).
 *
 * Inference contract is the same as the Python module: the caller
 * patch-embeds the latents (`patch_vae_latent`) and passes packed hidden
 * states [total_tokens, inner_dim] together with the matching rotary
 * embedding and per-sample cu_seqlens metadata. Batch size is 1. */

#ifndef BERNINI_TRANSFORMER_WAN_H
#define BERNINI_TRANSFORMER_WAN_H

#include "bt.h"
#include "safetensors.h"
#include "bjson.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bt_t *to_q_w, *to_q_b, *to_k_w, *to_k_b, *to_v_w, *to_v_b;
    bt_t *to_out_w, *to_out_b;
    bt_t *norm_q_w, *norm_k_w; /* rms_norm_across_heads over inner_dim */
} wan_attn_t;

typedef struct {
    wan_attn_t attn1, attn2;
    bt_t *norm2_w, *norm2_b;   /* FP32LayerNorm affine (cross_attn_norm) */
    bt_t *ffn0_w, *ffn0_b;     /* ffn.net.0.proj */
    bt_t *ffn2_w, *ffn2_b;     /* ffn.net.2 */
    bt_t *scale_shift_table;   /* [6, dim] */
} wan_block_t;

typedef struct {
    /* config (transformer_config.json / diffusers config.json) */
    int patch_t, patch_h, patch_w;
    int num_heads, head_dim, inner_dim;
    int in_channels, out_channels;
    int text_dim, freq_dim, ffn_dim, num_layers;
    int rope_max_seq_len;
    int use_src_id_rotary_emb;
    float eps;

    /* weights */
    bt_t *patch_emb_w, *patch_emb_b;     /* conv3d k=s=patch */
    bt_t *te_l1_w, *te_l1_b;             /* condition_embedder.time_embedder.linear_1 */
    bt_t *te_l2_w, *te_l2_b;
    bt_t *tp_w, *tp_b;                   /* condition_embedder.time_proj */
    bt_t *txt_l1_w, *txt_l1_b;           /* condition_embedder.text_embedder.linear_1 */
    bt_t *txt_l2_w, *txt_l2_b;
    wan_block_t *blocks;
    bt_t *proj_out_w, *proj_out_b;
    bt_t *scale_shift_table;             /* [2, dim] */

    /* RoPE tables: complex doubles [max_seq, head_dim/2] (t|h|w bands) and
     * the source-id table [max_seq, head_dim/2]. */
    double *rope_re, *rope_im;
    double *vid_re, *vid_im;
    int t_band, h_band, w_band;          /* complex counts per axis band */
} wan_transformer_t;

/* Tokens produced by patch_vae_latent: packed hidden states plus the rotary
 * table for those tokens (complex, [n_tokens, head_dim/2]). */
typedef struct {
    int64_t n_tokens;
    float  *hidden;   /* [n_tokens, inner_dim] (owned) */
    double *rope_re;  /* [n_tokens, head_dim/2] (owned) */
    double *rope_im;
} wan_tokens_t;

/* Load config from a parsed json object (fields of WanTransformer3DModel). */
int  wan_transformer_config(wan_transformer_t *m, const bj_value *cfg,
                            int use_src_id_rotary_emb);
/* Load weights from a safetensors store under `prefix` ("" for diffusers
 * layout). Returns 0 on success. */
int  wan_transformer_load(wan_transformer_t *m, const st_store *st, const char *prefix);
void wan_transformer_free(wan_transformer_t *m);

/* Patch-embed a VAE latent [C, T, H, W] into tokens with rotary embedding
 * (mirrors patch_vae_latent). */
void wan_patch_vae_latent(const wan_transformer_t *m, const float *latent,
                          int c, int t, int h, int w, int source_id, wan_tokens_t *out);
void wan_tokens_free(wan_tokens_t *tok);

/* Full forward over packed tokens (mirrors WanTransformer3DModel.forward with
 * batch 1, single sample): hidden [n_tokens, inner_dim] from concatenated
 * combos, text context [text_len, text_dim], timestep t.
 * Returns [n_tokens, out_channels * patch volume] (caller frees). */
float *wan_transformer_forward(const wan_transformer_t *m,
                               const float *hidden, int64_t n_tokens,
                               const double *rope_re, const double *rope_im,
                               const float *text, int64_t text_len, double timestep);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_TRANSFORMER_WAN_H */
