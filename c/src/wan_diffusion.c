/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "wan_diffusion.h"
#include "guidance.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------- packed <-> spatial ----------------------------- */

void wan_pack_latent(const float *sp, int c, int t, int h, int w, float *packed) {
    const int hh = h / 2, ww = w / 2;
    const int64_t thw = (int64_t)t * h * w, hw = (int64_t)h * w;
    const int pd = 4 * c;
#pragma omp parallel for collapse(2)
    for (int f = 0; f < t; f++)
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < ww; x++) {
                float *row = packed + (((int64_t)f * hh + y) * ww + x) * pd;
                for (int py = 0; py < 2; py++)
                    for (int px = 0; px < 2; px++)
                        for (int ci = 0; ci < c; ci++)
                            row[(py * 2 + px) * c + ci] =
                                sp[(int64_t)ci * thw + (int64_t)f * hw +
                                   (int64_t)(2 * y + py) * w + (2 * x + px)];
            }
}

void wan_unpack_latent(const float *packed, int c, int t, int h, int w, float *sp) {
    const int hh = h / 2, ww = w / 2;
    const int64_t thw = (int64_t)t * h * w, hw = (int64_t)h * w;
    const int pd = 4 * c;
#pragma omp parallel for collapse(2)
    for (int f = 0; f < t; f++)
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < ww; x++) {
                const float *row = packed + (((int64_t)f * hh + y) * ww + x) * pd;
                for (int py = 0; py < 2; py++)
                    for (int px = 0; px < 2; px++)
                        for (int ci = 0; ci < c; ci++)
                            sp[(int64_t)ci * thw + (int64_t)f * hw +
                               (int64_t)(2 * y + py) * w + (2 * x + px)] =
                                row[(py * 2 + px) * c + ci];
            }
}

/* ------------------------- token combo assembly --------------------------- */

typedef struct {
    float  *hidden;
    double *re, *im;
    int64_t n, cap;
    int     inner, half;
} combo_t;

static void combo_init(combo_t *cb, int inner, int half) {
    memset(cb, 0, sizeof(*cb));
    cb->inner = inner;
    cb->half = half;
}

static void combo_append(combo_t *cb, const wan_tokens_t *tok) {
    int64_t need = cb->n + tok->n_tokens;
    if (need > cb->cap) {
        cb->cap = need;
        cb->hidden = (float *)realloc(cb->hidden, sizeof(float) * (size_t)need * cb->inner);
        cb->re = (double *)realloc(cb->re, sizeof(double) * (size_t)need * cb->half);
        cb->im = (double *)realloc(cb->im, sizeof(double) * (size_t)need * cb->half);
    }
    memcpy(cb->hidden + cb->n * cb->inner, tok->hidden,
           sizeof(float) * (size_t)tok->n_tokens * cb->inner);
    memcpy(cb->re + cb->n * cb->half, tok->rope_re,
           sizeof(double) * (size_t)tok->n_tokens * cb->half);
    memcpy(cb->im + cb->n * cb->half, tok->rope_im,
           sizeof(double) * (size_t)tok->n_tokens * cb->half);
    cb->n += tok->n_tokens;
}

static void combo_reset(combo_t *cb) { cb->n = 0; }
static void combo_free(combo_t *cb) {
    free(cb->hidden);
    free(cb->re);
    free(cb->im);
    memset(cb, 0, sizeof(*cb));
}

/* Run the transformer over a combo and return the prediction restricted to
 * the trailing noisy tokens (mirrors `pred[:, msk, :]` with the noisy mask
 * being the last `n_noisy` positions). Caller frees. */
static float *fwd_combo(const wan_transformer_t *tr, const combo_t *cb,
                        int64_t n_noisy, const float *text, int64_t lt, double t) {
    float *pred = wan_transformer_forward(tr, cb->hidden, cb->n, cb->re, cb->im, text, lt, t);
    int out_dim = tr->out_channels * tr->patch_t * tr->patch_h * tr->patch_w;
    float *res = (float *)malloc(sizeof(float) * (size_t)n_noisy * out_dim);
    memcpy(res, pred + (cb->n - n_noisy) * out_dim, sizeof(float) * (size_t)n_noisy * out_dim);
    free(pred);
    return res;
}

/* --------------------------------- sample --------------------------------- */

bt_t *gen_wanx22_sample(gen_wanx22_t *g, const gen_sample_args_t *a) {
    /* scheduler: UniPC keeps the flow_shift it was constructed with; the
     * FlowMatch path takes the per-call shift override (matches Python). */
    bn_sched_set_timesteps(&g->scheduler, a->num_inference_steps,
                           g->use_unipc ? a->flow_shift : a->flow_shift);

    int num_frames = a->num_frames / g->vae_scale_factor_temporal * g->vae_scale_factor_temporal + 1;
    if (num_frames < 1) num_frames = 1;

    const wan_transformer_t *t1 = &g->transformer;
    const wan_transformer_t *t2 = g->has_transformer_2 ? &g->transformer_2 : &g->transformer;
    const int C = t1->in_channels;
    const int Tl = (num_frames - 1) / g->vae_scale_factor_temporal + 1;
    const int Hl = a->height / g->vae_scale_factor_spatial;
    const int Wl = a->width / g->vae_scale_factor_spatial;
    const int64_t n_noisy = (int64_t)Tl * (Hl / 2) * (Wl / 2);
    const int pd = 4 * C;

    double boundary = g->switch_dit_boundary * g->scheduler.num_train_timesteps;

    /* initial noise (spatial draw, then packed) */
    float *noise_sp = (float *)malloc(sizeof(float) * (size_t)C * Tl * Hl * Wl);
    bn_rng rng;
    bn_rng_seed(&rng, a->seed);
    bn_rng_randn(&rng, noise_sp, (int64_t)C * Tl * Hl * Wl);
    float *noisy = (float *)malloc(sizeof(float) * (size_t)n_noisy * pd);
    wan_pack_latent(noise_sp, C, Tl, Hl, Wl, noisy);
    free(noise_sp);

    const char *mode = a->guidance_mode;
    int is_rv2v = !strcmp(mode, "rv2v");
    int is_v2v = !strcmp(mode, "v2v");
    int is_v2v_chain = !strcmp(mode, "v2v_chain");
    int is_t2v = !strcmp(mode, "t2v");
    int is_r2v_apg = !strcmp(mode, "r2v_apg");
    int is_v2v_apg = !strcmp(mode, "v2v_apg");
    int is_t2v_apg = !strcmp(mode, "t2v_apg");
    if (!(is_rv2v || is_v2v || is_v2v_chain || is_t2v || is_r2v_apg || is_v2v_apg || is_t2v_apg)) {
        fprintf(stderr, "gen_wanx22: unknown guidance_mode '%s'\n", mode);
        free(noisy);
        return NULL;
    }

    bn_momentum_t mb1, mb2;
    bn_momentum_init(&mb1, a->momentum);
    bn_momentum_init(&mb2, a->momentum);
    double nt0 = a->norm_threshold[0];

    double omega_vid = a->omega_vid, omega_img = a->omega_img, omega_txt = a->omega_txt;
    int switched = 0;

    combo_t none_cb, v_cb, i_cb, vi_cb;
    combo_init(&none_cb, t1->inner_dim, t1->head_dim / 2);
    combo_init(&v_cb, t1->inner_dim, t1->head_dim / 2);
    combo_init(&i_cb, t1->inner_dim, t1->head_dim / 2);
    combo_init(&vi_cb, t1->inner_dim, t1->head_dim / 2);

    float *sp_buf = (float *)malloc(sizeof(float) * (size_t)C * Tl * Hl * Wl);
    int64_t packed_n = n_noisy * pd;

    for (int t_idx = 0; t_idx < g->scheduler.num_steps; t_idx++) {
        double t = g->scheduler.timesteps[t_idx];
        if (t < boundary && !switched && g->has_transformer_2) {
            switched = 1;
            omega_vid *= a->omega_scale;
            omega_img *= a->omega_scale;
            omega_txt *= a->omega_scale;
            fprintf(stderr, "gen_wanx22: switching to low-noise expert at step %d (t=%.1f)\n",
                    t_idx, t);
        }
        const wan_transformer_t *cur = switched ? t2 : t1;

        /* Build conditioning combos: each combo = condition tokens + the
         * shared noisy target latent (source_id 0). */
        combo_reset(&none_cb);
        combo_reset(&v_cb);
        combo_reset(&i_cb);
        combo_reset(&vi_cb);
        int sid = 1, sid_img = 1;

        for (int vi = 0; vi < a->n_videos; vi++) {
            bt_t *vl = a->video_latents[vi];
            wan_tokens_t tok;
            wan_patch_vae_latent(cur, vl->data, (int)vl->shape[0], (int)vl->shape[1],
                                 (int)vl->shape[2], (int)vl->shape[3], sid++, &tok);
            if (vi == 0) combo_append(&v_cb, &tok); /* only the first video joins V */
            combo_append(&vi_cb, &tok);
            wan_tokens_free(&tok);
        }
        for (int ii = 0; ii < a->n_images; ii++) {
            bt_t *il = a->image_latents[ii];
            wan_tokens_t tok;
            wan_patch_vae_latent(cur, il->data, (int)il->shape[0], (int)il->shape[1],
                                 (int)il->shape[2], (int)il->shape[3], sid++, &tok);
            combo_append(&vi_cb, &tok);
            wan_tokens_free(&tok);
            wan_patch_vae_latent(cur, il->data, (int)il->shape[0], (int)il->shape[1],
                                 (int)il->shape[2], (int)il->shape[3], sid_img++, &tok);
            combo_append(&i_cb, &tok);
            wan_tokens_free(&tok);
        }

        /* noisy target tokens (source_id 0), appended to every combo */
        wan_unpack_latent(noisy, C, Tl, Hl, Wl, sp_buf);
        wan_tokens_t ntok;
        wan_patch_vae_latent(cur, sp_buf, C, Tl, Hl, Wl, 0, &ntok);
        combo_append(&none_cb, &ntok);
        combo_append(&v_cb, &ntok);
        combo_append(&i_cb, &ntok);
        combo_append(&vi_cb, &ntok);
        wan_tokens_free(&ntok);

        const float *cond_text = a->prompt_embeds;
        int64_t cond_len = a->prompt_len;
        const float *uncond_text = a->uncond_embeds;
        int64_t uncond_len = a->uncond_len;

        float *noise_pred = NULL;

        if (is_rv2v) {
            /* eps = e0 + w_V(eV-e0) + w_I(eVI-eV) + w_TI(eVTI-eVI) */
            float *e0 = fwd_combo(cur, &none_cb, n_noisy, uncond_text, uncond_len, t);
            float *eV = fwd_combo(cur, &v_cb, n_noisy, uncond_text, uncond_len, t);
            float *eVI = fwd_combo(cur, &vi_cb, n_noisy, uncond_text, uncond_len, t);
            float *eVTI = fwd_combo(cur, &vi_cb, n_noisy, cond_text, cond_len, t);
            noise_pred = (float *)malloc(sizeof(float) * (size_t)packed_n);
            for (int64_t i = 0; i < packed_n; i++)
                noise_pred[i] = e0[i] + (float)omega_vid * (eV[i] - e0[i]) +
                                (float)omega_img * (eVI[i] - eV[i]) +
                                (float)omega_txt * (eVTI[i] - eVI[i]);
            free(e0); free(eV); free(eVI); free(eVTI);
        } else if (is_v2v) {
            float *eU = fwd_combo(cur, &vi_cb, n_noisy, uncond_text, uncond_len, t);
            float *eVTI = fwd_combo(cur, &vi_cb, n_noisy, cond_text, cond_len, t);
            noise_pred = (float *)malloc(sizeof(float) * (size_t)packed_n);
            for (int64_t i = 0; i < packed_n; i++)
                noise_pred[i] = eU[i] + (float)omega_txt * (eVTI[i] - eU[i]);
            free(eU); free(eVTI);
        } else if (is_v2v_chain) {
            float *e0 = fwd_combo(cur, &none_cb, n_noisy, uncond_text, uncond_len, t);
            float *eV = fwd_combo(cur, &v_cb, n_noisy, uncond_text, uncond_len, t);
            float *eVTI = fwd_combo(cur, &vi_cb, n_noisy, cond_text, cond_len, t);
            noise_pred = (float *)malloc(sizeof(float) * (size_t)packed_n);
            for (int64_t i = 0; i < packed_n; i++)
                noise_pred[i] = e0[i] + (float)omega_vid * (eV[i] - e0[i]) +
                                (float)omega_txt * (eVTI[i] - eV[i]);
            free(e0); free(eV); free(eVTI);
        } else if (is_t2v) {
            float *e0 = fwd_combo(cur, &none_cb, n_noisy, uncond_text, uncond_len, t);
            float *eT = fwd_combo(cur, &none_cb, n_noisy, cond_text, cond_len, t);
            noise_pred = (float *)malloc(sizeof(float) * (size_t)packed_n);
            for (int64_t i = 0; i < packed_n; i++)
                noise_pred[i] = e0[i] + (float)omega_txt * (eT[i] - e0[i]);
            free(e0); free(eT);
        } else {
            /* APG modes: convert v-pred to x-pred at the current sigma, run
             * normalized guidance in the spatial layout, convert back. */
            double sigma = bn_sched_apg_sigma(&g->scheduler, t_idx);
            int64_t sp_n = (int64_t)C * Tl * Hl * Wl;
            float *noisy_sp = (float *)malloc(sizeof(float) * (size_t)sp_n);
            wan_unpack_latent(noisy, C, Tl, Hl, Wl, noisy_sp);

            float *p1 = NULL, *p2 = NULL, *p3 = NULL;
            if (is_r2v_apg) {
                p1 = fwd_combo(cur, &none_cb, n_noisy, uncond_text, uncond_len, t);
                p2 = fwd_combo(cur, &i_cb, n_noisy, uncond_text, uncond_len, t);
                p3 = fwd_combo(cur, &i_cb, n_noisy, cond_text, cond_len, t);
            } else if (is_v2v_apg) {
                p1 = fwd_combo(cur, &vi_cb, n_noisy, uncond_text, uncond_len, t);
                p2 = fwd_combo(cur, &vi_cb, n_noisy, cond_text, cond_len, t);
            } else { /* t2v_apg */
                p1 = fwd_combo(cur, &none_cb, n_noisy, uncond_text, uncond_len, t);
                p2 = fwd_combo(cur, &none_cb, n_noisy, cond_text, cond_len, t);
            }
            /* x-preds in spatial layout: x = noisy - sigma * v */
            float *x1 = (float *)malloc(sizeof(float) * (size_t)sp_n);
            float *x2 = (float *)malloc(sizeof(float) * (size_t)sp_n);
            float *x3 = p3 ? (float *)malloc(sizeof(float) * (size_t)sp_n) : NULL;
            float *vtmp = (float *)malloc(sizeof(float) * (size_t)sp_n);
            wan_unpack_latent(p1, C, Tl, Hl, Wl, vtmp);
            for (int64_t i = 0; i < sp_n; i++) x1[i] = noisy_sp[i] - (float)sigma * vtmp[i];
            wan_unpack_latent(p2, C, Tl, Hl, Wl, vtmp);
            for (int64_t i = 0; i < sp_n; i++) x2[i] = noisy_sp[i] - (float)sigma * vtmp[i];
            if (p3) {
                wan_unpack_latent(p3, C, Tl, Hl, Wl, vtmp);
                for (int64_t i = 0; i < sp_n; i++) x3[i] = noisy_sp[i] - (float)sigma * vtmp[i];
            }

            float *x_guided = (float *)malloc(sizeof(float) * (size_t)sp_n);
            if (is_r2v_apg) {
                const float *preds[2] = {x2, x3};
                double scales[2] = {omega_img, omega_txt};
                bn_momentum_t mbs[2] = {mb1, mb2};
                bn_normalized_guidance_chain(x_guided, x1, preds, scales, mbs, a->eta,
                                             a->norm_threshold, 2, C, Tl, Hl, Wl);
                mb1 = mbs[0];
                mb2 = mbs[1];
            } else {
                bn_normalized_guidance(x_guided, x2, x1, omega_txt, &mb1, a->eta,
                                       nt0, C, Tl, Hl, Wl);
            }
            for (int64_t i = 0; i < sp_n; i++)
                vtmp[i] = (noisy_sp[i] - x_guided[i]) / (float)sigma;
            noise_pred = (float *)malloc(sizeof(float) * (size_t)packed_n);
            wan_pack_latent(vtmp, C, Tl, Hl, Wl, noise_pred);
            free(x1); free(x2); free(x3); free(vtmp); free(x_guided); free(noisy_sp);
            free(p1); free(p2); free(p3);
        }

        bn_sched_step(&g->scheduler, noise_pred, t, noisy, packed_n);
        free(noise_pred);
        fprintf(stderr, "\rgen_wanx22: step %d/%d", t_idx + 1, g->scheduler.num_steps);
    }
    fprintf(stderr, "\n");

    combo_free(&none_cb); combo_free(&v_cb); combo_free(&i_cb); combo_free(&vi_cb);
    bn_momentum_free(&mb1);
    bn_momentum_free(&mb2);
    free(sp_buf);

    bt_t *out = bt_new4(C, Tl, Hl, Wl);
    wan_unpack_latent(noisy, C, Tl, Hl, Wl, out->data);
    free(noisy);
    return out;
}

void gen_wanx22_free(gen_wanx22_t *g) {
    wan_transformer_free(&g->transformer);
    if (g->has_transformer_2) wan_transformer_free(&g->transformer_2);
    bn_sched_free(&g->scheduler);
}
