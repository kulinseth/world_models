/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "transformer_wan.h"
#include "attention.h"
#include "bjson.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROPE_THETA 10000.0

/* ----------------------------- config / init ----------------------------- */

int wan_transformer_config(wan_transformer_t *m, const bj_value *cfg,
                           int use_src_id_rotary_emb) {
    memset(m, 0, sizeof(*m));
    int patch[3] = {1, 2, 2};
    bj_get_iarray(cfg, "patch_size", patch, 3);
    m->patch_t = patch[0];
    m->patch_h = patch[1];
    m->patch_w = patch[2];
    m->num_heads = (int)bj_get_num(cfg, "num_attention_heads", 40);
    m->head_dim = (int)bj_get_num(cfg, "attention_head_dim", 128);
    m->inner_dim = m->num_heads * m->head_dim;
    m->in_channels = (int)bj_get_num(cfg, "in_channels", 16);
    m->out_channels = (int)bj_get_num(cfg, "out_channels", 16);
    m->text_dim = (int)bj_get_num(cfg, "text_dim", 4096);
    m->freq_dim = (int)bj_get_num(cfg, "freq_dim", 256);
    m->ffn_dim = (int)bj_get_num(cfg, "ffn_dim", 13824);
    m->num_layers = (int)bj_get_num(cfg, "num_layers", 40);
    m->rope_max_seq_len = (int)bj_get_num(cfg, "rope_max_seq_len", 1024);
    m->use_src_id_rotary_emb = use_src_id_rotary_emb ||
                               bj_get_bool(cfg, "use_src_id_rotary_emb", 0);
    m->eps = (float)bj_get_num(cfg, "eps", 1e-6);

    /* RoPE bands (WanRotaryPosEmbed): h_dim = w_dim = 2*(hd//6), t the rest */
    int hd = m->head_dim;
    int h_dim = 2 * (hd / 6);
    int t_dim = hd - 2 * h_dim;
    m->t_band = t_dim / 2;
    m->h_band = h_dim / 2;
    m->w_band = h_dim / 2;

    int half = hd / 2;
    int max_seq = m->rope_max_seq_len;
    m->rope_re = (double *)malloc(sizeof(double) * (size_t)max_seq * half);
    m->rope_im = (double *)malloc(sizeof(double) * (size_t)max_seq * half);
    /* per-band frequencies, concatenated t|h|w along the complex axis
     * (matches get_1d_rotary_pos_embed(dim, ..., use_real=False)) */
    for (int pos = 0; pos < max_seq; pos++) {
        int off = 0;
        const int band_dims[3] = {t_dim, h_dim, h_dim};
        for (int b = 0; b < 3; b++) {
            int bc = band_dims[b] / 2;
            for (int j = 0; j < bc; j++) {
                double freq = 1.0 / pow(ROPE_THETA, (double)(2 * j) / (double)band_dims[b]);
                double ang = (double)pos * freq;
                m->rope_re[(size_t)pos * half + off + j] = cos(ang);
                m->rope_im[(size_t)pos * half + off + j] = sin(ang);
            }
            off += bc;
        }
    }
    if (m->use_src_id_rotary_emb) {
        m->vid_re = (double *)malloc(sizeof(double) * (size_t)max_seq * half);
        m->vid_im = (double *)malloc(sizeof(double) * (size_t)max_seq * half);
        for (int pos = 0; pos < max_seq; pos++) {
            for (int j = 0; j < half; j++) {
                double freq = 1.0 / pow(ROPE_THETA, (double)(2 * j) / (double)hd);
                double ang = (double)pos * freq;
                m->vid_re[(size_t)pos * half + j] = cos(ang);
                m->vid_im[(size_t)pos * half + j] = sin(ang);
            }
        }
    }
    return 0;
}

static int load_attn(wan_attn_t *a, const st_store *st, const char *prefix,
                     const char *block, const char *attn) {
    char p[256];
#define GET(field, name)                                                       \
    do {                                                                       \
        snprintf(p, sizeof(p), "%s.%s.%s", block, attn, name);                 \
        a->field = st_get_pfx(st, prefix, "%s", p);                            \
        if (!a->field) return -1;                                              \
    } while (0)
    GET(to_q_w, "to_q.weight"); GET(to_q_b, "to_q.bias");
    GET(to_k_w, "to_k.weight"); GET(to_k_b, "to_k.bias");
    GET(to_v_w, "to_v.weight"); GET(to_v_b, "to_v.bias");
    GET(to_out_w, "to_out.0.weight"); GET(to_out_b, "to_out.0.bias");
    GET(norm_q_w, "norm_q.weight"); GET(norm_k_w, "norm_k.weight");
#undef GET
    return 0;
}

int wan_transformer_load(wan_transformer_t *m, const st_store *st, const char *prefix) {
#define GET(field, ...)                                                        \
    do {                                                                       \
        m->field = st_get_pfx(st, prefix, __VA_ARGS__);                        \
        if (!m->field) return -1;                                              \
    } while (0)
    GET(patch_emb_w, "patch_embedding.weight");
    GET(patch_emb_b, "patch_embedding.bias");
    GET(te_l1_w, "condition_embedder.time_embedder.linear_1.weight");
    GET(te_l1_b, "condition_embedder.time_embedder.linear_1.bias");
    GET(te_l2_w, "condition_embedder.time_embedder.linear_2.weight");
    GET(te_l2_b, "condition_embedder.time_embedder.linear_2.bias");
    GET(tp_w, "condition_embedder.time_proj.weight");
    GET(tp_b, "condition_embedder.time_proj.bias");
    GET(txt_l1_w, "condition_embedder.text_embedder.linear_1.weight");
    GET(txt_l1_b, "condition_embedder.text_embedder.linear_1.bias");
    GET(txt_l2_w, "condition_embedder.text_embedder.linear_2.weight");
    GET(txt_l2_b, "condition_embedder.text_embedder.linear_2.bias");
    GET(proj_out_w, "proj_out.weight");
    GET(proj_out_b, "proj_out.bias");
    GET(scale_shift_table, "scale_shift_table");

    m->blocks = (wan_block_t *)calloc((size_t)m->num_layers, sizeof(wan_block_t));
    char blk[64];
    for (int i = 0; i < m->num_layers; i++) {
        wan_block_t *b = &m->blocks[i];
        snprintf(blk, sizeof(blk), "blocks.%d", i);
        if (load_attn(&b->attn1, st, prefix, blk, "attn1")) return -1;
        if (load_attn(&b->attn2, st, prefix, blk, "attn2")) return -1;
        b->norm2_w = st_get_pfx(st, prefix, "%s.norm2.weight", blk);
        b->norm2_b = st_get_pfx(st, prefix, "%s.norm2.bias", blk);
        b->ffn0_w = st_get_pfx(st, prefix, "%s.ffn.net.0.proj.weight", blk);
        b->ffn0_b = st_get_pfx(st, prefix, "%s.ffn.net.0.proj.bias", blk);
        b->ffn2_w = st_get_pfx(st, prefix, "%s.ffn.net.2.weight", blk);
        b->ffn2_b = st_get_pfx(st, prefix, "%s.ffn.net.2.bias", blk);
        b->scale_shift_table = st_get_pfx(st, prefix, "%s.scale_shift_table", blk);
        if (!b->norm2_w || !b->norm2_b || !b->ffn0_w || !b->ffn0_b ||
            !b->ffn2_w || !b->ffn2_b || !b->scale_shift_table)
            return -1;
        fprintf(stderr, "\rwan: loaded block %d/%d", i + 1, m->num_layers);
    }
    fprintf(stderr, "\n");
#undef GET
    return 0;
}

static void free_attn(wan_attn_t *a) {
    bt_free(a->to_q_w); bt_free(a->to_q_b); bt_free(a->to_k_w); bt_free(a->to_k_b);
    bt_free(a->to_v_w); bt_free(a->to_v_b); bt_free(a->to_out_w); bt_free(a->to_out_b);
    bt_free(a->norm_q_w); bt_free(a->norm_k_w);
}

void wan_transformer_free(wan_transformer_t *m) {
    bt_free(m->patch_emb_w); bt_free(m->patch_emb_b);
    bt_free(m->te_l1_w); bt_free(m->te_l1_b); bt_free(m->te_l2_w); bt_free(m->te_l2_b);
    bt_free(m->tp_w); bt_free(m->tp_b);
    bt_free(m->txt_l1_w); bt_free(m->txt_l1_b); bt_free(m->txt_l2_w); bt_free(m->txt_l2_b);
    bt_free(m->proj_out_w); bt_free(m->proj_out_b); bt_free(m->scale_shift_table);
    if (m->blocks) {
        for (int i = 0; i < m->num_layers; i++) {
            wan_block_t *b = &m->blocks[i];
            free_attn(&b->attn1); free_attn(&b->attn2);
            bt_free(b->norm2_w); bt_free(b->norm2_b);
            bt_free(b->ffn0_w); bt_free(b->ffn0_b); bt_free(b->ffn2_w); bt_free(b->ffn2_b);
            bt_free(b->scale_shift_table);
        }
        free(m->blocks);
    }
    free(m->rope_re); free(m->rope_im); free(m->vid_re); free(m->vid_im);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------ rope / patch ------------------------------ */

void wan_tokens_free(wan_tokens_t *tok) {
    free(tok->hidden);
    free(tok->rope_re);
    free(tok->rope_im);
    memset(tok, 0, sizeof(*tok));
}

void wan_patch_vae_latent(const wan_transformer_t *m, const float *latent,
                          int c, int t, int h, int w, int source_id, wan_tokens_t *out) {
    const int pt = m->patch_t, ph = m->patch_h, pw = m->patch_w;
    const int ppf = t / pt, pph = h / ph, ppw = w / pw;
    const int64_t n = (int64_t)ppf * pph * ppw;
    const int half = m->head_dim / 2;
    const int inner = m->inner_dim;

    out->n_tokens = n;
    out->hidden = (float *)malloc(sizeof(float) * (size_t)n * inner);
    out->rope_re = (double *)malloc(sizeof(double) * (size_t)n * half);
    out->rope_im = (double *)malloc(sizeof(double) * (size_t)n * half);

    /* patch embedding: conv3d with kernel == stride == patch_size, then
     * flatten to token-major layout [n_tokens, inner_dim] */
    const int64_t thw = (int64_t)t * h * w, hw = (int64_t)h * w;
    const int pvol = pt * ph * pw;
#pragma omp parallel for collapse(2)
    for (int f = 0; f < ppf; f++) {
        for (int y = 0; y < pph; y++) {
            for (int x = 0; x < ppw; x++) {
                int64_t tok = ((int64_t)f * pph + y) * ppw + x;
                float *hrow = out->hidden + tok * inner;
                for (int d = 0; d < inner; d++) {
                    const float *wd = m->patch_emb_w->data + (int64_t)d * c * pvol;
                    double acc = (double)m->patch_emb_b->data[d];
                    for (int ic = 0; ic < c; ic++) {
                        const float *ib = latent + (int64_t)ic * thw;
                        const float *wc = wd + (int64_t)ic * pvol;
                        for (int dt = 0; dt < pt; dt++)
                            for (int dy = 0; dy < ph; dy++)
                                for (int dx = 0; dx < pw; dx++)
                                    acc += (double)ib[(int64_t)(f * pt + dt) * hw +
                                                      (int64_t)(y * ph + dy) * w +
                                                      (x * pw + dx)] *
                                           (double)wc[(dt * ph + dy) * pw + dx];
                    }
                    hrow[d] = (float)acc;
                }
            }
        }
    }

    /* rotary embedding: t|h|w bands by token coordinate, optionally rotated by
     * the per-source-id frequency (use_src_id_rotary_emb) */
    for (int f = 0; f < ppf; f++) {
        for (int y = 0; y < pph; y++) {
            for (int x = 0; x < ppw; x++) {
                int64_t tok = ((int64_t)f * pph + y) * ppw + x;
                double *rre = out->rope_re + tok * half;
                double *rim = out->rope_im + tok * half;
                const double *tb_re = m->rope_re + (size_t)f * half;
                const double *tb_im = m->rope_im + (size_t)f * half;
                const double *hb_re = m->rope_re + (size_t)y * half;
                const double *hb_im = m->rope_im + (size_t)y * half;
                const double *wb_re = m->rope_re + (size_t)x * half;
                const double *wb_im = m->rope_im + (size_t)x * half;
                for (int j = 0; j < m->t_band; j++) { rre[j] = tb_re[j]; rim[j] = tb_im[j]; }
                for (int j = 0; j < m->h_band; j++) {
                    rre[m->t_band + j] = hb_re[m->t_band + j];
                    rim[m->t_band + j] = hb_im[m->t_band + j];
                }
                for (int j = 0; j < m->w_band; j++) {
                    rre[m->t_band + m->h_band + j] = wb_re[m->t_band + m->h_band + j];
                    rim[m->t_band + m->h_band + j] = wb_im[m->t_band + m->h_band + j];
                }
                if (m->use_src_id_rotary_emb) {
                    const double *vre = m->vid_re + (size_t)source_id * half;
                    const double *vim = m->vid_im + (size_t)source_id * half;
                    for (int j = 0; j < half; j++) {
                        double re = rre[j] * vre[j] - rim[j] * vim[j];
                        double im = rre[j] * vim[j] + rim[j] * vre[j];
                        rre[j] = re;
                        rim[j] = im;
                    }
                }
            }
        }
    }
}

/* x: [n, heads, head_dim] viewed flat; complex multiply per (token, pair),
 * broadcast over heads; computed in double like the fp64 path in Python. */
static void apply_rotary(float *x, int64_t n, int heads, int hd,
                         const double *rope_re, const double *rope_im) {
    const int half = hd / 2;
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        const double *rre = rope_re + i * half;
        const double *rim = rope_im + i * half;
        float *base = x + i * (int64_t)heads * hd;
        for (int h = 0; h < heads; h++) {
            float *p = base + (int64_t)h * hd;
            for (int j = 0; j < half; j++) {
                double a = (double)p[2 * j], b = (double)p[2 * j + 1];
                p[2 * j] = (float)(a * rre[j] - b * rim[j]);
                p[2 * j + 1] = (float)(a * rim[j] + b * rre[j]);
            }
        }
    }
}

/* ------------------------------- forward ---------------------------------- */

static float *sinusoidal_timestep(double t, int dim) {
    /* diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0 */
    int half = dim / 2;
    float *emb = (float *)malloc(sizeof(float) * (size_t)dim);
    for (int j = 0; j < half; j++) {
        double freq = exp(-log(10000.0) * (double)j / (double)half);
        double a = t * freq;
        emb[j] = (float)cos(a);
        emb[half + j] = (float)sin(a);
    }
    return emb;
}

float *wan_transformer_forward(const wan_transformer_t *m,
                               const float *hidden_in, int64_t n,
                               const double *rope_re, const double *rope_im,
                               const float *text, int64_t lt, double timestep) {
    const int dim = m->inner_dim;
    const int heads = m->num_heads, hd = m->head_dim;
    const float eps = m->eps;

    /* ---- condition embedder ---- */
    float *sin_emb = sinusoidal_timestep(timestep, m->freq_dim);
    float *temb = (float *)malloc(sizeof(float) * (size_t)dim);
    float *tmp = (float *)malloc(sizeof(float) * (size_t)dim);
    bt_linear(sin_emb, m->te_l1_w->data, m->te_l1_b->data, tmp, 1, m->freq_dim, dim);
    bt_silu(tmp, dim);
    bt_linear(tmp, m->te_l2_w->data, m->te_l2_b->data, temb, 1, dim, dim);
    /* timestep_proj = time_proj(silu(temb)) -> [6, dim] */
    memcpy(tmp, temb, sizeof(float) * (size_t)dim);
    bt_silu(tmp, dim);
    float *tproj = (float *)malloc(sizeof(float) * (size_t)6 * dim);
    bt_linear(tmp, m->tp_w->data, m->tp_b->data, tproj, 1, dim, 6 * dim);
    free(tmp);
    free(sin_emb);

    /* text context: PixArtAlphaTextProjection (linear, gelu_tanh, linear) */
    float *txt_mid = (float *)malloc(sizeof(float) * (size_t)lt * dim);
    bt_linear(text, m->txt_l1_w->data, m->txt_l1_b->data, txt_mid, lt, m->text_dim, dim);
    bt_gelu_tanh(txt_mid, lt * (int64_t)dim);
    float *txt_ctx = (float *)malloc(sizeof(float) * (size_t)lt * dim);
    bt_linear(txt_mid, m->txt_l2_w->data, m->txt_l2_b->data, txt_ctx, lt, dim, dim);
    free(txt_mid);

    float *x = (float *)malloc(sizeof(float) * (size_t)n * dim);
    memcpy(x, hidden_in, sizeof(float) * (size_t)n * dim);

    float *norm_buf = (float *)malloc(sizeof(float) * (size_t)n * dim);
    float *q = (float *)malloc(sizeof(float) * (size_t)n * dim);
    float *k = (float *)malloc(sizeof(float) * (size_t)(n > lt ? n : lt) * dim);
    float *v = (float *)malloc(sizeof(float) * (size_t)(n > lt ? n : lt) * dim);
    float *attn = (float *)malloc(sizeof(float) * (size_t)n * dim);
    float *proj = (float *)malloc(sizeof(float) * (size_t)n * dim);
    int64_t ffn_rows = n;
    float *ffn_buf = (float *)malloc(sizeof(float) * (size_t)ffn_rows * m->ffn_dim);

    int64_t cu_self[2] = {0, n};
    int64_t cu_q2[2] = {0, n};
    int64_t cu_k2[2] = {0, lt};
    float *mod = (float *)malloc(sizeof(float) * 6 * (size_t)dim);

    for (int layer = 0; layer < m->num_layers; layer++) {
        const wan_block_t *b = &m->blocks[layer];
        /* six modulation vectors: scale_shift_table + timestep_proj */
        const float *sst = b->scale_shift_table->data;
        const float *e[6];
        for (int i = 0; i < 6; i++) {
            e[i] = mod + i * dim;
            for (int d = 0; d < dim; d++) mod[i * dim + d] = sst[i * dim + d] + tproj[i * dim + d];
        }

        /* 1. self-attention */
        memcpy(norm_buf, x, sizeof(float) * (size_t)n * dim);
        bt_layernorm(norm_buf, n, dim, NULL, NULL, eps);
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            for (int d = 0; d < dim; d++)
                norm_buf[i * dim + d] = norm_buf[i * dim + d] * (1.0f + e[1][d]) + e[0][d];

        bt_linear(norm_buf, b->attn1.to_q_w->data, b->attn1.to_q_b->data, q, n, dim, dim);
        bt_linear(norm_buf, b->attn1.to_k_w->data, b->attn1.to_k_b->data, k, n, dim, dim);
        bt_linear(norm_buf, b->attn1.to_v_w->data, b->attn1.to_v_b->data, v, n, dim, dim);
        bt_rmsnorm(q, n, dim, b->attn1.norm_q_w->data, eps);
        bt_rmsnorm(k, n, dim, b->attn1.norm_k_w->data, eps);
        apply_rotary(q, n, heads, hd, rope_re, rope_im);
        apply_rotary(k, n, heads, hd, rope_re, rope_im);
        bn_varlen_attention(q, k, v, attn, cu_self, cu_self, 1, heads, hd, 0);
        bt_linear(attn, b->attn1.to_out_w->data, b->attn1.to_out_b->data, proj, n, dim, dim);
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            for (int d = 0; d < dim; d++)
                x[i * dim + d] += proj[i * dim + d] * e[2][d];

        /* 2. cross-attention */
        memcpy(norm_buf, x, sizeof(float) * (size_t)n * dim);
        bt_layernorm(norm_buf, n, dim, b->norm2_w->data, b->norm2_b->data, eps);
        bt_linear(norm_buf, b->attn2.to_q_w->data, b->attn2.to_q_b->data, q, n, dim, dim);
        bt_linear(txt_ctx, b->attn2.to_k_w->data, b->attn2.to_k_b->data, k, lt, dim, dim);
        bt_linear(txt_ctx, b->attn2.to_v_w->data, b->attn2.to_v_b->data, v, lt, dim, dim);
        bt_rmsnorm(q, n, dim, b->attn2.norm_q_w->data, eps);
        bt_rmsnorm(k, lt, dim, b->attn2.norm_k_w->data, eps);
        bn_varlen_attention(q, k, v, attn, cu_q2, cu_k2, 1, heads, hd, 0);
        bt_linear(attn, b->attn2.to_out_w->data, b->attn2.to_out_b->data, proj, n, dim, dim);
        bt_add(x, proj, n * (int64_t)dim);

        /* 3. feed-forward */
        memcpy(norm_buf, x, sizeof(float) * (size_t)n * dim);
        bt_layernorm(norm_buf, n, dim, NULL, NULL, eps);
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            for (int d = 0; d < dim; d++)
                norm_buf[i * dim + d] = norm_buf[i * dim + d] * (1.0f + e[4][d]) + e[3][d];
        bt_linear(norm_buf, b->ffn0_w->data, b->ffn0_b->data, ffn_buf, n, dim, m->ffn_dim);
        bt_gelu_tanh(ffn_buf, n * (int64_t)m->ffn_dim);
        bt_linear(ffn_buf, b->ffn2_w->data, b->ffn2_b->data, proj, n, m->ffn_dim, dim);
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            for (int d = 0; d < dim; d++)
                x[i * dim + d] += proj[i * dim + d] * e[5][d];
    }

    /* output norm + projection: shift/scale = scale_shift_table + temb */
    const float *out_sst = m->scale_shift_table->data;
    bt_layernorm(x, n, dim, NULL, NULL, eps);
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        for (int d = 0; d < dim; d++) {
            float shift = out_sst[d] + temb[d];
            float scale = out_sst[dim + d] + temb[d];
            x[i * dim + d] = x[i * dim + d] * (1.0f + scale) + shift;
        }
    int out_dim = m->out_channels * m->patch_t * m->patch_h * m->patch_w;
    float *out = (float *)malloc(sizeof(float) * (size_t)n * out_dim);
    bt_linear(x, m->proj_out_w->data, m->proj_out_b->data, out, n, dim, out_dim);

    free(x); free(norm_buf); free(q); free(k); free(v);
    free(attn); free(proj); free(ffn_buf); free(mod);
    free(txt_ctx); free(temb); free(tproj);
    return out;
}
