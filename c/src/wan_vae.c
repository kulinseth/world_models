/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "wan_vae.h"
#include "bjson.h"
#include "safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CACHE_T 2

/* ----------------------------- small helpers ------------------------------ */

static bt_t *frame_slice(const bt_t *x, int t0, int n) {
    int C = (int)x->shape[0], T = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    (void)T;
    bt_t *out = bt_new4(C, n, H, W);
    int64_t hw = (int64_t)H * W;
    for (int c = 0; c < C; c++)
        memcpy(out->data + ((int64_t)c * n) * hw,
               x->data + ((int64_t)c * x->shape[1] + t0) * hw,
               sizeof(float) * (size_t)n * hw);
    return out;
}

static bt_t *frames_tail(const bt_t *x, int n) {
    int T = (int)x->shape[1];
    int take = n < T ? n : T;
    return frame_slice(x, T - take, take);
}

static bt_t *frames_cat(const bt_t *a, const bt_t *b) {
    int C = (int)a->shape[0], Ta = (int)a->shape[1], Tb = (int)b->shape[1];
    int H = (int)a->shape[2], W = (int)a->shape[3];
    bt_t *out = bt_new4(C, Ta + Tb, H, W);
    int64_t hw = (int64_t)H * W;
    for (int c = 0; c < C; c++) {
        memcpy(out->data + ((int64_t)c * (Ta + Tb)) * hw,
               a->data + ((int64_t)c * Ta) * hw, sizeof(float) * (size_t)Ta * hw);
        memcpy(out->data + ((int64_t)c * (Ta + Tb) + Ta) * hw,
               b->data + ((int64_t)c * Tb) * hw, sizeof(float) * (size_t)Tb * hw);
    }
    return out;
}

/* WanRMS_norm with channel_first: x / max(||x||_C, 1e-12) * sqrt(C) * gamma */
static void rms_channel(bt_t *x, const bt_t *gamma) {
    int C = (int)x->shape[0];
    int64_t thw = x->numel / C;
    double scale = sqrt((double)C);
#pragma omp parallel for
    for (int64_t i = 0; i < thw; i++) {
        double ss = 0.0;
        for (int c = 0; c < C; c++) {
            float v = x->data[(int64_t)c * thw + i];
            ss += (double)v * (double)v;
        }
        double nrm = sqrt(ss);
        if (nrm < 1e-12) nrm = 1e-12;
        double inv = scale / nrm;
        for (int c = 0; c < C; c++)
            x->data[(int64_t)c * thw + i] =
                (float)((double)x->data[(int64_t)c * thw + i] * inv * (double)gamma->data[c]);
    }
}

static void silu_t(bt_t *x) { bt_silu(x->data, x->numel); }

/* Causal conv pattern shared by conv_in / resblock convs / conv_out:
 * compute the next cache from the raw input, run the conv against the old
 * cache (which reduces the front zero padding), store the new cache. */
static bt_t *cconv3_cached(const vconv_t *cv, const bt_t *x, vcache_t *cache, int *idx) {
    vcache_t *slot = &cache[(*idx)++];
    int cout = (int)cv->w->shape[0], cin = (int)cv->w->shape[1];
    int kt = (int)cv->w->shape[2], kh = (int)cv->w->shape[3], kw = (int)cv->w->shape[4];
    int pt = kt / 2, ph = kh / 2, pw = kw / 2;
    (void)cin;

    bt_t *newc = frames_tail(x, CACHE_T);
    if (newc->shape[1] < CACHE_T && slot->state == 1) {
        bt_t *last = frames_tail(slot->x, 1);
        bt_t *aug = frames_cat(last, newc);
        bt_free(last);
        bt_free(newc);
        newc = aug;
    }

    const bt_t *in = x;
    bt_t *joined = NULL;
    int cached = 0;
    if (slot->state == 1 && pt > 0) {
        joined = frames_cat(slot->x, x);
        in = joined;
        cached = (int)slot->x->shape[1];
    }
    int front = 2 * pt - cached;
    if (front < 0) front = 0;

    int C = (int)in->shape[0], T = (int)in->shape[1], H = (int)in->shape[2], W = (int)in->shape[3];
    int ot, oh, ow;
    int oT = (T + front - kt) + 1; /* stride 1 */
    bt_t *out = bt_new4(cout, oT, (H + 2 * ph - kh) + 1, (W + 2 * pw - kw) + 1);
    bt_conv3d(in->data, C, T, H, W, cv->w->data, cv->b ? cv->b->data : NULL, cout,
              kt, kh, kw, 1, 1, 1, front, 0, ph, pw, out->data, &ot, &oh, &ow);
    if (joined) bt_free(joined);

    if (slot->x) bt_free(slot->x);
    slot->x = newc;
    slot->state = 1;
    return out;
}

/* plain (uncached) causal conv: zero pad 2*pt in front */
static bt_t *cconv3_plain(const vconv_t *cv, const bt_t *x) {
    int cout = (int)cv->w->shape[0];
    int kt = (int)cv->w->shape[2], kh = (int)cv->w->shape[3], kw = (int)cv->w->shape[4];
    int pt = kt / 2, ph = kh / 2, pw = kw / 2;
    int C = (int)x->shape[0], T = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    int ot, oh, ow;
    bt_t *out = bt_new4(cout, (T + 2 * pt - kt) + 1, (H + 2 * ph - kh) + 1, (W + 2 * pw - kw) + 1);
    bt_conv3d(x->data, C, T, H, W, cv->w->data, cv->b ? cv->b->data : NULL, cout,
              kt, kh, kw, 1, 1, 1, 2 * pt, 0, ph, pw, out->data, &ot, &oh, &ow);
    return out;
}

/* ------------------------------ blocks ------------------------------------ */

static bt_t *vres_fwd(const vres_t *r, bt_t *x, vcache_t *cache, int *idx) {
    bt_t *h;
    if (r->has_shortcut)
        h = cconv3_plain(&r->shortcut, x);
    else
        h = bt_clone(x);

    rms_channel(x, r->norm1_g);
    silu_t(x);
    bt_t *y = cconv3_cached(&r->conv1, x, cache, idx);
    bt_free(x);
    rms_channel(y, r->norm2_g);
    silu_t(y);
    bt_t *z = cconv3_cached(&r->conv2, y, cache, idx);
    bt_free(y);
    bt_add(z->data, h->data, z->numel);
    bt_free(h);
    return z;
}

/* single-head spatial attention per frame */
static bt_t *vattn_fwd(const vattn_t *a, bt_t *x) {
    int C = (int)x->shape[0], T = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    int64_t hw = (int64_t)H * W;
    float scale = 1.0f / sqrtf((float)C);
    bt_t *out = bt_clone(x);

    bt_t *frame = bt_new3(C, H, W);
    float *qkv = (float *)malloc(sizeof(float) * (size_t)3 * C * hw);
    float *scores = (float *)malloc(sizeof(float) * (size_t)hw);
    float *res = (float *)malloc(sizeof(float) * (size_t)C * hw);

    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++)
            memcpy(frame->data + (int64_t)c * hw,
                   x->data + ((int64_t)c * T + t) * hw, sizeof(float) * (size_t)hw);
        rms_channel(frame, a->norm_g);
        /* 1x1 convs == per-pixel linear over channels */
        int oh, ow;
        bt_conv2d(frame->data, C, H, W, a->qkv.w->data, a->qkv.b ? a->qkv.b->data : NULL,
                  3 * C, 1, 1, 1, 1, 0, 0, 0, 0, qkv, &oh, &ow);
        const float *q = qkv, *k = qkv + (int64_t)C * hw, *v = qkv + (int64_t)2 * C * hw;
        /* tokens are pixels, dim C (layout [C, HW] -> gather per pixel) */
#pragma omp parallel for
        for (int64_t i = 0; i < hw; i++) {
            float *sc = (float *)malloc(sizeof(float) * (size_t)hw);
            float mx = -1e30f;
            for (int64_t j = 0; j < hw; j++) {
                double dot = 0.0;
                for (int c = 0; c < C; c++)
                    dot += (double)q[(int64_t)c * hw + i] * (double)k[(int64_t)c * hw + j];
                sc[j] = (float)(dot * scale);
                if (sc[j] > mx) mx = sc[j];
            }
            double sum = 0.0;
            for (int64_t j = 0; j < hw; j++) {
                sc[j] = expf(sc[j] - mx);
                sum += sc[j];
            }
            float inv = (float)(1.0 / sum);
            for (int c = 0; c < C; c++) {
                double acc = 0.0;
                for (int64_t j = 0; j < hw; j++)
                    acc += (double)sc[j] * (double)v[(int64_t)c * hw + j];
                res[(int64_t)c * hw + i] = (float)(acc * inv);
            }
            free(sc);
        }
        int oh2, ow2;
        bt_conv2d(res, C, H, W, a->proj.w->data, a->proj.b ? a->proj.b->data : NULL,
                  C, 1, 1, 1, 1, 0, 0, 0, 0, qkv, &oh2, &ow2);
        for (int c = 0; c < C; c++)
            bt_add(out->data + ((int64_t)c * T + t) * hw, qkv + (int64_t)c * hw, hw);
        (void)scores;
    }
    free(qkv);
    free(scores);
    free(res);
    bt_free(frame);
    bt_free(x);
    return out;
}

/* spatial resample applied per frame */
static bt_t *spatial_resample(const vresample_t *rs, const bt_t *x) {
    int C = (int)x->shape[0], T = (int)x->shape[1], H = (int)x->shape[2], W = (int)x->shape[3];
    int64_t hw = (int64_t)H * W;
    if (rs->mode == 0) return bt_clone(x);

    int up = (rs->mode == 1 || rs->mode == 2);
    int cout = (int)rs->conv.w->shape[0];
    int oh, ow;
    bt_t *out = NULL;
    float *frame = (float *)malloc(sizeof(float) * (size_t)C * hw);
    float *upbuf = up ? (float *)malloc(sizeof(float) * (size_t)C * 4 * hw) : NULL;

    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++)
            memcpy(frame + (int64_t)c * hw, x->data + ((int64_t)c * T + t) * hw,
                   sizeof(float) * (size_t)hw);
        float *conv_out;
        if (up) {
            bt_upsample_nearest2x(frame, C, H, W, upbuf);
            conv_out = (float *)malloc(sizeof(float) * (size_t)cout * (2 * H) * (2 * W));
            bt_conv2d(upbuf, C, 2 * H, 2 * W, rs->conv.w->data, rs->conv.b->data, cout,
                      3, 3, 1, 1, 1, 1, 1, 1, conv_out, &oh, &ow);
        } else {
            /* ZeroPad2d((0,1,0,1)) + conv 3x3 stride 2 */
            conv_out = (float *)malloc(sizeof(float) * (size_t)cout * ((H + 1) / 2) * ((W + 1) / 2));
            bt_conv2d(frame, C, H, W, rs->conv.w->data, rs->conv.b->data, cout,
                      3, 3, 2, 2, 0, 1, 0, 1, conv_out, &oh, &ow);
        }
        if (!out) out = bt_new4(cout, T, oh, ow);
        for (int c = 0; c < cout; c++)
            memcpy(out->data + ((int64_t)c * T + t) * (int64_t)oh * ow,
                   conv_out + (int64_t)c * oh * ow, sizeof(float) * (size_t)oh * ow);
        free(conv_out);
    }
    free(frame);
    free(upbuf);
    return out;
}

static bt_t *vresample_fwd(const vresample_t *rs, bt_t *x, vcache_t *cache, int *idx) {
    int C = (int)x->shape[0], H = (int)x->shape[2], W = (int)x->shape[3];

    if (rs->mode == 2) { /* upsample3d: temporal doubling via time_conv */
        vcache_t *slot = &cache[(*idx)++];
        if (slot->state == 0) {
            slot->state = 2; /* "Rep": first chunk passes through untouched */
        } else {
            bt_t *newc = frames_tail(x, CACHE_T);
            if (newc->shape[1] < CACHE_T) {
                if (slot->state == 1) {
                    bt_t *last = frames_tail(slot->x, 1);
                    bt_t *aug = frames_cat(last, newc);
                    bt_free(last);
                    bt_free(newc);
                    newc = aug;
                } else { /* "Rep": prepend zeros */
                    bt_t *zeros = bt_new4((int)newc->shape[0], 1, (int)newc->shape[2],
                                          (int)newc->shape[3]);
                    memset(zeros->data, 0, sizeof(float) * (size_t)zeros->numel);
                    bt_t *aug = frames_cat(zeros, newc);
                    bt_free(zeros);
                    bt_free(newc);
                    newc = aug;
                }
            }
            /* conv (3,1,1) pad (1,0,0): zero-padded for "Rep", cached otherwise */
            const bt_t *in = x;
            bt_t *joined = NULL;
            int cached = 0;
            if (slot->state == 1) {
                joined = frames_cat(slot->x, x);
                in = joined;
                cached = (int)slot->x->shape[1];
            }
            int front = 2 - cached;
            int cout = (int)rs->time_conv.w->shape[0]; /* == 2*C */
            int ot, oh, ow;
            bt_t *tc = bt_new4(cout, (int)in->shape[1] + front - 3 + 1, H, W);
            bt_conv3d(in->data, C, (int)in->shape[1], H, W, rs->time_conv.w->data,
                      rs->time_conv.b->data, cout, 3, 1, 1, 1, 1, 1, front, 0, 0, 0,
                      tc->data, &ot, &oh, &ow);
            if (joined) bt_free(joined);
            if (slot->state == 1) bt_free(slot->x);
            slot->x = newc;
            slot->state = 1;
            /* interleave: out[c][2t] = tc[c][t], out[c][2t+1] = tc[C+c][t] */
            bt_t *dbl = bt_new4(C, 2 * ot, H, W);
            int64_t hw = (int64_t)H * W;
            for (int c = 0; c < C; c++)
                for (int t = 0; t < ot; t++) {
                    memcpy(dbl->data + ((int64_t)c * 2 * ot + 2 * t) * hw,
                           tc->data + ((int64_t)c * ot + t) * hw, sizeof(float) * (size_t)hw);
                    memcpy(dbl->data + ((int64_t)c * 2 * ot + 2 * t + 1) * hw,
                           tc->data + ((int64_t)(C + c) * ot + t) * hw,
                           sizeof(float) * (size_t)hw);
                }
            bt_free(tc);
            bt_free(x);
            x = dbl;
        }
    }

    bt_t *sp = spatial_resample(rs, x);
    bt_free(x);
    x = sp;

    if (rs->mode == 4) { /* downsample3d: strided time conv with 1-frame cache */
        vcache_t *slot = &cache[(*idx)++];
        if (slot->state == 0) {
            slot->x = bt_clone(x);
            slot->state = 1;
        } else {
            bt_t *newc = frames_tail(x, 1);
            bt_t *last = frames_tail(slot->x, 1);
            bt_t *in = frames_cat(last, x);
            bt_free(last);
            int cout = (int)rs->time_conv.w->shape[0];
            int Ti = (int)in->shape[1];
            int ot, oh, ow;
            bt_t *tc = bt_new4(cout, (Ti - 3) / 2 + 1, (int)x->shape[2], (int)x->shape[3]);
            bt_conv3d(in->data, (int)in->shape[0], Ti, (int)in->shape[2], (int)in->shape[3],
                      rs->time_conv.w->data, rs->time_conv.b->data, cout,
                      3, 1, 1, 2, 1, 1, 0, 0, 0, 0, tc->data, &ot, &oh, &ow);
            bt_free(in);
            bt_free(slot->x);
            slot->x = newc;
            bt_free(x);
            x = tc;
        }
    }
    return x;
}

static bt_t *mid_fwd(const vres_t *res, const vattn_t *attn, bt_t *x,
                     vcache_t *cache, int *idx) {
    x = vres_fwd(&res[0], x, cache, idx);
    x = vattn_fwd(attn, x);
    x = vres_fwd(&res[1], x, cache, idx);
    return x;
}

/* ------------------------------ load --------------------------------------- */

static int load_conv(vconv_t *c, const st_store *st, const char *fmt, ...) {
    char name[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    char key[560];
    snprintf(key, sizeof(key), "%s.weight", name);
    c->w = st_get(st, key);
    snprintf(key, sizeof(key), "%s.bias", name);
    c->b = st_get(st, key);
    if (!c->w) {
        fprintf(stderr, "wan_vae: missing %s.weight\n", name);
        return -1;
    }
    return 0;
}

static int load_res(vres_t *r, const st_store *st, const char *base) {
    char key[560];
    snprintf(key, sizeof(key), "%s.norm1.gamma", base);
    r->norm1_g = st_get(st, key);
    snprintf(key, sizeof(key), "%s.norm2.gamma", base);
    r->norm2_g = st_get(st, key);
    if (load_conv(&r->conv1, st, "%s.conv1", base)) return -1;
    if (load_conv(&r->conv2, st, "%s.conv2", base)) return -1;
    r->in_dim = (int)r->conv1.w->shape[1];
    r->out_dim = (int)r->conv1.w->shape[0];
    r->has_shortcut = 0;
    snprintf(key, sizeof(key), "%s.conv_shortcut.weight", base);
    if (st_has(st, key)) {
        r->has_shortcut = 1;
        if (load_conv(&r->shortcut, st, "%s.conv_shortcut", base)) return -1;
    }
    return r->norm1_g && r->norm2_g ? 0 : -1;
}

static int load_attn_block(vattn_t *a, const st_store *st, const char *base) {
    char key[560];
    snprintf(key, sizeof(key), "%s.norm.gamma", base);
    a->norm_g = st_get(st, key);
    if (load_conv(&a->qkv, st, "%s.to_qkv", base)) return -1;
    if (load_conv(&a->proj, st, "%s.proj", base)) return -1;
    a->dim = (int)a->proj.w->shape[0];
    return a->norm_g ? 0 : -1;
}

static int load_resample(vresample_t *rs, const st_store *st, const char *base, int mode) {
    rs->mode = mode;
    if (load_conv(&rs->conv, st, "%s.resample.1", base)) return -1;
    if (mode == 2 || mode == 4) {
        if (load_conv(&rs->time_conv, st, "%s.time_conv", base)) return -1;
    }
    return 0;
}

int wan_vae_load(wan_vae_t *v, const char *dir) {
    memset(v, 0, sizeof(*v));
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    bj_value *cfg = bj_parse_file(path);
    if (!cfg) {
        fprintf(stderr, "wan_vae: missing %s\n", path);
        return -1;
    }
    v->base_dim = (int)bj_get_num(cfg, "base_dim", 96);
    v->z_dim = (int)bj_get_num(cfg, "z_dim", 16);
    v->num_res_blocks = (int)bj_get_num(cfg, "num_res_blocks", 2);
    v->n_mult = (int)bj_get_iarray(cfg, "dim_mult", v->dim_mult, 4);
    if (!v->n_mult) {
        v->n_mult = 4;
        v->dim_mult[0] = 1; v->dim_mult[1] = 2; v->dim_mult[2] = 4; v->dim_mult[3] = 4;
    }
    int tds[4] = {0, 1, 1, 0};
    size_t ntd = bj_get_iarray(cfg, "temperal_downsample", tds, 4);
    if (!ntd) { tds[0] = 0; tds[1] = 1; tds[2] = 1; }
    for (int i = 0; i < 4; i++) v->temperal_downsample[i] = tds[i];
    bj_get_farray(cfg, "latents_mean", v->latents_mean, WAN_VAE_MAX_Z);
    bj_get_farray(cfg, "latents_std", v->latents_std, WAN_VAE_MAX_Z);
    if (bj_get_bool(cfg, "is_residual", 0)) {
        fprintf(stderr, "wan_vae: is_residual (Wan2.2 TI2V) VAE is not supported by the C port\n");
        bj_free(cfg);
        return -1;
    }
    bj_free(cfg);

    st_store *st = st_open_dir(dir);
    if (!st) {
        fprintf(stderr, "wan_vae: no safetensors in %s\n", dir);
        return -1;
    }

    int rc = 0;
    rc |= load_conv(&v->enc_conv_in, st, "encoder.conv_in");
    /* encoder down_blocks: flattened module list of res blocks + resamples */
    int li = 0, k = 0;
    char base[256];
    for (int i = 0; i < v->n_mult; i++) {
        for (int rb = 0; rb < v->num_res_blocks; rb++) {
            snprintf(base, sizeof(base), "encoder.down_blocks.%d", k++);
            v->enc_layers[li].kind = 0;
            rc |= load_res(&v->enc_layers[li].res, st, base);
            li++;
        }
        if (i != v->n_mult - 1) {
            snprintf(base, sizeof(base), "encoder.down_blocks.%d", k++);
            v->enc_layers[li].kind = 1;
            rc |= load_resample(&v->enc_layers[li].rs, st, base,
                                v->temperal_downsample[i] ? 4 : 3);
            li++;
        }
    }
    v->n_enc_layers = li;
    rc |= load_res(&v->enc_mid_res[0], st, "encoder.mid_block.resnets.0");
    rc |= load_attn_block(&v->enc_mid_attn, st, "encoder.mid_block.attentions.0");
    rc |= load_res(&v->enc_mid_res[1], st, "encoder.mid_block.resnets.1");
    v->enc_norm_out_g = st_get(st, "encoder.norm_out.gamma");
    rc |= load_conv(&v->enc_conv_out, st, "encoder.conv_out");
    rc |= load_conv(&v->quant_conv, st, "quant_conv");
    rc |= load_conv(&v->post_quant_conv, st, "post_quant_conv");

    rc |= load_conv(&v->dec_conv_in, st, "decoder.conv_in");
    rc |= load_res(&v->dec_mid_res[0], st, "decoder.mid_block.resnets.0");
    rc |= load_attn_block(&v->dec_mid_attn, st, "decoder.mid_block.attentions.0");
    rc |= load_res(&v->dec_mid_res[1], st, "decoder.mid_block.resnets.1");
    /* decoder up blocks: temperal_upsample = reversed(temperal_downsample) */
    v->n_up_blocks = v->n_mult;
    for (int i = 0; i < v->n_mult; i++) {
        vup_block_t *ub = &v->up_blocks[i];
        ub->n_res = v->num_res_blocks + 1;
        for (int rb = 0; rb < ub->n_res; rb++) {
            snprintf(base, sizeof(base), "decoder.up_blocks.%d.resnets.%d", i, rb);
            rc |= load_res(&ub->res[rb], st, base);
        }
        int up_flag = i != v->n_mult - 1;
        ub->has_upsampler = up_flag;
        if (up_flag) {
            int t_up = v->temperal_downsample[v->n_mult - 2 - i]; /* reversed list */
            snprintf(base, sizeof(base), "decoder.up_blocks.%d.upsamplers.0", i);
            rc |= load_resample(&ub->up, st, base, t_up ? 2 : 1);
        }
    }
    v->dec_norm_out_g = st_get(st, "decoder.norm_out.gamma");
    rc |= load_conv(&v->dec_conv_out, st, "decoder.conv_out");
    st_close(st);
    if (rc || !v->enc_norm_out_g || !v->dec_norm_out_g) {
        fprintf(stderr, "wan_vae: failed to load weights from %s\n", dir);
        return -1;
    }
    fprintf(stderr, "wan_vae: loaded (base_dim=%d z_dim=%d)\n", v->base_dim, v->z_dim);
    return 0;
}

static void free_conv(vconv_t *c) { bt_free(c->w); bt_free(c->b); }
static void free_res(vres_t *r) {
    bt_free(r->norm1_g); bt_free(r->norm2_g);
    free_conv(&r->conv1); free_conv(&r->conv2);
    if (r->has_shortcut) free_conv(&r->shortcut);
}

static void clear_cache(wan_vae_t *v) {
    for (int i = 0; i < WAN_VAE_MAX_CACHE; i++) {
        if (v->cache[i].state == 1) bt_free(v->cache[i].x);
        v->cache[i].state = 0;
        v->cache[i].x = NULL;
    }
}

void wan_vae_free(wan_vae_t *v) {
    clear_cache(v);
    free_conv(&v->enc_conv_in);
    for (int i = 0; i < v->n_enc_layers; i++) {
        if (v->enc_layers[i].kind == 0) free_res(&v->enc_layers[i].res);
        else {
            free_conv(&v->enc_layers[i].rs.conv);
            if (v->enc_layers[i].rs.mode == 2 || v->enc_layers[i].rs.mode == 4)
                free_conv(&v->enc_layers[i].rs.time_conv);
        }
    }
    free_res(&v->enc_mid_res[0]); free_res(&v->enc_mid_res[1]);
    bt_free(v->enc_mid_attn.norm_g); free_conv(&v->enc_mid_attn.qkv); free_conv(&v->enc_mid_attn.proj);
    bt_free(v->enc_norm_out_g);
    free_conv(&v->enc_conv_out); free_conv(&v->quant_conv); free_conv(&v->post_quant_conv);
    free_conv(&v->dec_conv_in);
    free_res(&v->dec_mid_res[0]); free_res(&v->dec_mid_res[1]);
    bt_free(v->dec_mid_attn.norm_g); free_conv(&v->dec_mid_attn.qkv); free_conv(&v->dec_mid_attn.proj);
    for (int i = 0; i < v->n_up_blocks; i++) {
        for (int rb = 0; rb < v->up_blocks[i].n_res; rb++) free_res(&v->up_blocks[i].res[rb]);
        if (v->up_blocks[i].has_upsampler) {
            free_conv(&v->up_blocks[i].up.conv);
            if (v->up_blocks[i].up.mode == 2) free_conv(&v->up_blocks[i].up.time_conv);
        }
    }
    bt_free(v->dec_norm_out_g);
    free_conv(&v->dec_conv_out);
    memset(v, 0, sizeof(*v));
}

/* ----------------------------- encode / decode ----------------------------- */

static bt_t *encoder_chunk(wan_vae_t *v, bt_t *x, int *idx) {
    x = cconv3_cached(&v->enc_conv_in, x, v->cache, idx);
    for (int i = 0; i < v->n_enc_layers; i++) {
        if (v->enc_layers[i].kind == 0)
            x = vres_fwd(&v->enc_layers[i].res, x, v->cache, idx);
        else
            x = vresample_fwd(&v->enc_layers[i].rs, x, v->cache, idx);
    }
    x = mid_fwd(v->enc_mid_res, &v->enc_mid_attn, x, v->cache, idx);
    rms_channel(x, v->enc_norm_out_g);
    silu_t(x);
    bt_t *out = cconv3_cached(&v->enc_conv_out, x, v->cache, idx);
    bt_free(x);
    return out;
}

bt_t *wan_vae_encode(wan_vae_t *v, const bt_t *pixels) {
    clear_cache(v);
    int T = (int)pixels->shape[1];
    int iters = 1 + (T - 1) / 4;
    bt_t *out = NULL;
    for (int i = 0; i < iters; i++) {
        int idx = 0;
        bt_t *chunk = (i == 0) ? frame_slice(pixels, 0, 1)
                               : frame_slice(pixels, 1 + 4 * (i - 1), 4);
        bt_t *enc = encoder_chunk(v, chunk, &idx);
        bt_free(chunk);
        if (!out) out = enc;
        else {
            bt_t *cat = frames_cat(out, enc);
            bt_free(out);
            bt_free(enc);
            out = cat;
        }
        fprintf(stderr, "\rwan_vae: encode chunk %d/%d", i + 1, iters);
    }
    fprintf(stderr, "\n");
    bt_t *q = cconv3_plain(&v->quant_conv, out);
    bt_free(out);
    clear_cache(v);
    /* DiagonalGaussianDistribution.mode() == mean == first z channels */
    bt_t *mean = bt_new4(v->z_dim, (int)q->shape[1], (int)q->shape[2], (int)q->shape[3]);
    memcpy(mean->data, q->data, sizeof(float) * (size_t)mean->numel);
    bt_free(q);
    return mean;
}

static bt_t *decoder_chunk(wan_vae_t *v, bt_t *x, int *idx) {
    x = cconv3_cached(&v->dec_conv_in, x, v->cache, idx);
    x = mid_fwd(v->dec_mid_res, &v->dec_mid_attn, x, v->cache, idx);
    for (int i = 0; i < v->n_up_blocks; i++) {
        vup_block_t *ub = &v->up_blocks[i];
        for (int rb = 0; rb < ub->n_res; rb++)
            x = vres_fwd(&ub->res[rb], x, v->cache, idx);
        if (ub->has_upsampler)
            x = vresample_fwd(&ub->up, x, v->cache, idx);
    }
    rms_channel(x, v->dec_norm_out_g);
    silu_t(x);
    bt_t *out = cconv3_cached(&v->dec_conv_out, x, v->cache, idx);
    bt_free(x);
    return out;
}

bt_t *wan_vae_decode(wan_vae_t *v, const bt_t *latents) {
    clear_cache(v);
    bt_t *x = cconv3_plain(&v->post_quant_conv, latents);
    int T = (int)x->shape[1];
    bt_t *out = NULL;
    for (int i = 0; i < T; i++) {
        int idx = 0;
        bt_t *chunk = frame_slice(x, i, 1);
        bt_t *dec = decoder_chunk(v, chunk, &idx);
        bt_free(chunk);
        if (!out) out = dec;
        else {
            bt_t *cat = frames_cat(out, dec);
            bt_free(out);
            bt_free(dec);
            out = cat;
        }
        fprintf(stderr, "\rwan_vae: decode frame %d/%d", i + 1, T);
    }
    fprintf(stderr, "\n");
    bt_free(x);
    clear_cache(v);
    bt_clamp(out->data, -1.0f, 1.0f, out->numel);
    return out;
}
