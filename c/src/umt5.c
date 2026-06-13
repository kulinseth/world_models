/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "umt5.h"
#include "attention.h"
#include "bjson.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int umt5_load(umt5_t *m, const char *dir) {
    memset(m, 0, sizeof(*m));
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    bj_value *cfg = bj_parse_file(path);
    if (!cfg) {
        fprintf(stderr, "umt5: missing %s\n", path);
        return -1;
    }
    m->d_model = (int)bj_get_num(cfg, "d_model", 4096);
    m->d_kv = (int)bj_get_num(cfg, "d_kv", 64);
    m->d_ff = (int)bj_get_num(cfg, "d_ff", 10240);
    m->num_layers = (int)bj_get_num(cfg, "num_layers", 24);
    m->num_heads = (int)bj_get_num(cfg, "num_heads", 64);
    m->rel_buckets = (int)bj_get_num(cfg, "relative_attention_num_buckets", 32);
    m->rel_max_distance = (int)bj_get_num(cfg, "relative_attention_max_distance", 128);
    m->eps = (float)bj_get_num(cfg, "layer_norm_epsilon", 1e-6);
    bj_free(cfg);

    st_store *st = st_open_dir(dir);
    if (!st) {
        fprintf(stderr, "umt5: no safetensors found in %s\n", dir);
        return -1;
    }
    m->shared = st_get(st, "shared.weight");
    if (!m->shared) m->shared = st_get(st, "encoder.embed_tokens.weight");
    m->final_ln_w = st_get(st, "encoder.final_layer_norm.weight");
    if (!m->shared || !m->final_ln_w) {
        fprintf(stderr, "umt5: missing embeddings / final layer norm\n");
        st_close(st);
        return -1;
    }
    m->blocks = (umt5_block_t *)calloc((size_t)m->num_layers, sizeof(umt5_block_t));
    for (int i = 0; i < m->num_layers; i++) {
        umt5_block_t *b = &m->blocks[i];
        b->ln1_w = st_get_pfx(st, "", "encoder.block.%d.layer.0.layer_norm.weight", i);
        b->q_w = st_get_pfx(st, "", "encoder.block.%d.layer.0.SelfAttention.q.weight", i);
        b->k_w = st_get_pfx(st, "", "encoder.block.%d.layer.0.SelfAttention.k.weight", i);
        b->v_w = st_get_pfx(st, "", "encoder.block.%d.layer.0.SelfAttention.v.weight", i);
        b->o_w = st_get_pfx(st, "", "encoder.block.%d.layer.0.SelfAttention.o.weight", i);
        b->rel_bias = st_get_pfx(st, "",
            "encoder.block.%d.layer.0.SelfAttention.relative_attention_bias.weight", i);
        b->ln2_w = st_get_pfx(st, "", "encoder.block.%d.layer.1.layer_norm.weight", i);
        b->wi0_w = st_get_pfx(st, "", "encoder.block.%d.layer.1.DenseReluDense.wi_0.weight", i);
        b->wi1_w = st_get_pfx(st, "", "encoder.block.%d.layer.1.DenseReluDense.wi_1.weight", i);
        b->wo_w = st_get_pfx(st, "", "encoder.block.%d.layer.1.DenseReluDense.wo.weight", i);
        if (!b->ln1_w || !b->q_w || !b->k_w || !b->v_w || !b->o_w || !b->rel_bias ||
            !b->ln2_w || !b->wi0_w || !b->wi1_w || !b->wo_w) {
            st_close(st);
            return -1;
        }
        fprintf(stderr, "\rumt5: loaded block %d/%d", i + 1, m->num_layers);
    }
    fprintf(stderr, "\n");
    st_close(st);
    return 0;
}

static void free_block(umt5_block_t *b) {
    bt_free(b->ln1_w); bt_free(b->q_w); bt_free(b->k_w); bt_free(b->v_w);
    bt_free(b->o_w); bt_free(b->rel_bias); bt_free(b->ln2_w);
    bt_free(b->wi0_w); bt_free(b->wi1_w); bt_free(b->wo_w);
}

void umt5_free(umt5_t *m) {
    bt_free(m->shared);
    bt_free(m->final_ln_w);
    if (m->blocks)
        for (int i = 0; i < m->num_layers; i++) free_block(&m->blocks[i]);
    free(m->blocks);
    memset(m, 0, sizeof(*m));
}

/* T5 relative position bucket (bidirectional). */
static int rel_bucket(int relative_position, int num_buckets, int max_distance) {
    int ret = 0;
    num_buckets /= 2;
    if (relative_position > 0) ret += num_buckets;
    int n = abs(relative_position);
    int max_exact = num_buckets / 2;
    if (n < max_exact) return ret + n;
    int large = max_exact +
                (int)(log((double)n / max_exact) / log((double)max_distance / max_exact) *
                      (num_buckets - max_exact));
    if (large > num_buckets - 1) large = num_buckets - 1;
    return ret + large;
}

float *umt5_encode(const umt5_t *m, const int32_t *ids, const int32_t *mask, int len) {
    const int d = m->d_model, H = m->num_heads, dk = m->d_kv;
    const int inner = H * dk;

    float *x = (float *)malloc(sizeof(float) * (size_t)len * d);
    for (int i = 0; i < len; i++)
        memcpy(x + (int64_t)i * d, m->shared->data + (int64_t)ids[i] * d, sizeof(float) * (size_t)d);

    /* additive key mask: 0 where attended, -inf-ish where padded */
    float *key_mask = (float *)malloc(sizeof(float) * (size_t)len);
    for (int i = 0; i < len; i++) key_mask[i] = mask && !mask[i] ? -1e9f : 0.0f;

    /* per-layer position bias [H, len, len] */
    float *bias = (float *)malloc(sizeof(float) * (size_t)H * len * len);
    int *buckets = (int *)malloc(sizeof(int) * (size_t)len * len);
    for (int q = 0; q < len; q++)
        for (int k = 0; k < len; k++)
            buckets[q * len + k] = rel_bucket(k - q, m->rel_buckets, m->rel_max_distance);

    float *h = (float *)malloc(sizeof(float) * (size_t)len * d);
    float *q = (float *)malloc(sizeof(float) * (size_t)len * inner);
    float *k = (float *)malloc(sizeof(float) * (size_t)len * inner);
    float *v = (float *)malloc(sizeof(float) * (size_t)len * inner);
    float *attn = (float *)malloc(sizeof(float) * (size_t)len * inner);
    float *proj = (float *)malloc(sizeof(float) * (size_t)len * d);
    float *ff0 = (float *)malloc(sizeof(float) * (size_t)len * m->d_ff);
    float *ff1 = (float *)malloc(sizeof(float) * (size_t)len * m->d_ff);

    for (int layer = 0; layer < m->num_layers; layer++) {
        const umt5_block_t *b = &m->blocks[layer];

        /* ---- self-attention ---- */
        memcpy(h, x, sizeof(float) * (size_t)len * d);
        bt_rmsnorm(h, len, d, b->ln1_w->data, m->eps);
        bt_linear(h, b->q_w->data, NULL, q, len, d, inner);
        bt_linear(h, b->k_w->data, NULL, k, len, d, inner);
        bt_linear(h, b->v_w->data, NULL, v, len, d, inner);

        /* this layer's relative position bias */
        const float *rb = b->rel_bias->data; /* [buckets, H] */
        for (int hd = 0; hd < H; hd++)
            for (int qq = 0; qq < len; qq++)
                for (int kk = 0; kk < len; kk++)
                    bias[((int64_t)hd * len + qq) * len + kk] =
                        rb[buckets[qq * len + kk] * H + hd];

        /* T5 attention has no 1/sqrt(d) scaling */
        bn_attention_bias(q, k, v, attn, len, len, H, dk, bias, key_mask, 1.0f);
        bt_linear(attn, b->o_w->data, NULL, proj, len, inner, d);
        bt_add(x, proj, (int64_t)len * d);

        /* ---- gated feed-forward ---- */
        memcpy(h, x, sizeof(float) * (size_t)len * d);
        bt_rmsnorm(h, len, d, b->ln2_w->data, m->eps);
        bt_linear(h, b->wi0_w->data, NULL, ff0, len, d, m->d_ff);
        bt_gelu_tanh(ff0, (int64_t)len * m->d_ff); /* gelu_new */
        bt_linear(h, b->wi1_w->data, NULL, ff1, len, d, m->d_ff);
        bt_mul(ff0, ff1, (int64_t)len * m->d_ff);
        bt_linear(ff0, b->wo_w->data, NULL, proj, len, m->d_ff, d);
        bt_add(x, proj, (int64_t)len * d);
    }

    bt_rmsnorm(x, len, d, m->final_ln_w->data, m->eps);

    free(h); free(q); free(k); free(v); free(attn); free(proj);
    free(ff0); free(ff1); free(bias); free(buckets); free(key_mask);
    return x;
}
