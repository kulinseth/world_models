/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* End-to-end Bernini Renderer inference pipeline:
 * preprocess -> sample -> decode -> save (mirrors bernini/pipeline.py). */

#include "../include/bernini.h"

#include "bjson.h"
#include "bt.h"
#include "data_utils.h"
#include "media.h"
#include "safetensors.h"
#include "scheduler.h"
#include "tokenizer.h"
#include "transformer_wan.h"
#include "umt5.h"
#include "wan_diffusion.h"
#include "wan_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_SEQ_LEN 512

/* weights.py prefix candidates */
static const char *const HIGH_NOISE_PREFIXES[] = {"diff_dec.transformer.", "transformer.", ""};
static const char *const LOW_NOISE_PREFIXES[] = {"diff_dec.transformer_2.", "transformer_2.", ""};

struct bernini_renderer_pipeline {
    gen_wanx22_t diff_dec;
    wan_vae_t vae;
    umt5_t t5;
    bn_tokenizer *tokenizer;
    int max_sequence_length;
    double switch_dit_boundary;
};

/* ------------------------------ helpers ----------------------------------- */

static int dir_exists(const char *p) {
    struct stat sb;
    return p && stat(p, &sb) == 0 && S_ISDIR(sb.st_mode);
}

/* _prompt_clean: collapse whitespace runs and trim. (The Python version also
 * applies ftfy + html unescaping; clean UTF-8 input is assumed here.) */
static char *prompt_clean(const char *sys, const char *text) {
    size_t cap = strlen(sys) + strlen(text) + 2;
    char *joined = (char *)malloc(cap);
    snprintf(joined, cap, "%s%s", sys, text);
    char *out = (char *)malloc(cap);
    size_t o = 0;
    int in_space = 1; /* leading trim */
    for (size_t i = 0; joined[i]; i++) {
        unsigned char c = (unsigned char)joined[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!in_space) out[o++] = ' ';
            in_space = 1;
        } else {
            out[o++] = (char)c;
            in_space = 0;
        }
    }
    while (o > 0 && out[o - 1] == ' ') o--;
    out[o] = '\0';
    free(joined);
    return out;
}

/* encode_prompt (renderer.py): T5 encode, keep seq_len rows, zero-pad to max */
static float *encode_prompt(struct bernini_renderer_pipeline *p, const char *text, int *len_out) {
    int32_t ids[MAX_SEQ_LEN], mask[MAX_SEQ_LEN];
    int real = bn_tokenize(p->tokenizer, text, ids, mask, MAX_SEQ_LEN);
    float *hidden = umt5_encode(&p->t5, ids, mask, MAX_SEQ_LEN);
    int d = p->t5.d_model;
    /* zero out positions past seq_len */
    for (int i = real; i < p->max_sequence_length; i++)
        memset(hidden + (int64_t)i * d, 0, sizeof(float) * (size_t)d);
    *len_out = p->max_sequence_length;
    return hidden;
}

/* _vae_encode: encode pixels [3,T,H,W] and normalize with latents mean/std */
static bt_t *vae_encode_norm(struct bernini_renderer_pipeline *p, const bt_t *pixels) {
    bt_t *lat = wan_vae_encode(&p->vae, pixels);
    int z = (int)lat->shape[0];
    int64_t thw = lat->numel / z;
    for (int c = 0; c < z; c++) {
        float mu = p->vae.latents_mean[c], sd = p->vae.latents_std[c];
        float *ptr = lat->data + (int64_t)c * thw;
        for (int64_t i = 0; i < thw; i++) ptr[i] = (ptr[i] - mu) / sd;
    }
    return lat;
}

/* preprocess_image: load, VAE transform, [3,1,H,W] */
static bt_t *preprocess_image(const char *path, int max_image_size) {
    bn_image_t img;
    if (bn_image_load(path, &img)) {
        fprintf(stderr, "pipeline: cannot load image %s\n", path);
        return NULL;
    }
    bt_t *chw = bn_vae_transform(img.rgb, img.width, img.height, max_image_size, 1, 16);
    bn_image_free(&img);
    bt_t *out = bt_new4(3, 1, chw->shape[1], chw->shape[2]);
    memcpy(out->data, chw->data, sizeof(float) * (size_t)chw->numel);
    bt_free(chw);
    return out;
}

/* preprocess_video: sample frames at `fps`, VAE transform, [3,T,H,W] */
static bt_t *preprocess_video(const char *path, int fps, int max_image_size, int max_image_num) {
    int total, w, h;
    double vfps;
    if (bn_video_probe(path, &total, &vfps, &w, &h)) {
        fprintf(stderr, "pipeline: cannot probe video %s\n", path);
        return NULL;
    }
    int n_idx;
    int *idx = bn_smart_video_nframes(total, vfps, fps, 4, max_image_num, 1, &n_idx);
    bn_video_t vid;
    if (bn_video_load(path, idx, n_idx, &vid)) {
        free(idx);
        fprintf(stderr, "pipeline: cannot decode video %s\n", path);
        return NULL;
    }
    free(idx);
    bt_t *out = NULL;
    for (int i = 0; i < vid.n_frames; i++) {
        bt_t *chw = bn_vae_transform(vid.rgb + (size_t)i * vid.width * vid.height * 3,
                                     vid.width, vid.height, max_image_size, 1, 16);
        if (!out) out = bt_new4(3, vid.n_frames, chw->shape[1], chw->shape[2]);
        int64_t hw = chw->shape[1] * chw->shape[2];
        for (int c = 0; c < 3; c++)
            memcpy(out->data + ((int64_t)c * vid.n_frames + i) * hw,
                   chw->data + (int64_t)c * hw, sizeof(float) * (size_t)hw);
        bt_free(chw);
    }
    bn_video_free(&vid);
    return out;
}

/* --------------------------- from_pretrained ------------------------------- */

bernini_renderer_pipeline_t *bernini_renderer_from_pretrained(
    const char *config_dir, const char *high_noise_ckpt, const char *low_noise_ckpt,
    int use_unipc, double shift_override, int use_src_id_rotary_emb) {

    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", config_dir);
    bj_value *cfg = bj_parse_file(path);
    if (!cfg) {
        fprintf(stderr, "pipeline: cannot read %s\n", path);
        return NULL;
    }

    struct bernini_renderer_pipeline *p =
        (struct bernini_renderer_pipeline *)calloc(1, sizeof(*p));

    const char *wan22_base = bj_get_str(cfg, "wan22_base", NULL);
    char base_buf[4096];
    /* _prefer_local_dir: fall back to config_dir when it holds the components */
    if (!dir_exists(wan22_base)) {
        snprintf(base_buf, sizeof(base_buf), "%s/tokenizer", config_dir);
        char b2[4096], b3[4096];
        snprintf(b2, sizeof(b2), "%s/text_encoder", config_dir);
        snprintf(b3, sizeof(b3), "%s/vae", config_dir);
        if (dir_exists(base_buf) && dir_exists(b2) && dir_exists(b3)) {
            wan22_base = config_dir;
        }
    }
    if (!wan22_base || !dir_exists(wan22_base)) {
        fprintf(stderr,
                "pipeline: wan22_base '%s' not found locally. The C port needs a local "
                "directory (download the HF repo first).\n",
                wan22_base ? wan22_base : "(null)");
        bj_free(cfg);
        free(p);
        return NULL;
    }
    snprintf(base_buf, sizeof(base_buf), "%s", wan22_base);
    wan22_base = base_buf;

    p->switch_dit_boundary = bj_get_num(cfg, "switch_dit_boundary", 0.875);
    p->max_sequence_length = (int)bj_get_num(cfg, "max_sequence_length", 512);
    double shift = bj_get_num(cfg, "shift", 3.0);
    int cfg_unipc = bj_get_bool(cfg, "use_unipc", 1);
    int cfg_srcid = bj_get_bool(cfg, "use_src_id_rotary_emb", 1);
    int skip_t2 = bj_get_bool(cfg, "skip_transformer_2", 0);
    bj_free(cfg);
    if (use_unipc >= 0) cfg_unipc = use_unipc;
    if (shift_override > 0) shift = shift_override;
    if (use_src_id_rotary_emb >= 0) cfg_srcid = use_src_id_rotary_emb;

    /* tokenizer */
    snprintf(path, sizeof(path), "%s/tokenizer/spiece.model", wan22_base);
    p->tokenizer = bn_tokenizer_load(path);
    if (!p->tokenizer) goto fail;

    /* text encoder */
    snprintf(path, sizeof(path), "%s/text_encoder", wan22_base);
    if (umt5_load(&p->t5, path)) goto fail;

    /* vae (fp32, mirrors AutoencoderKLWan.from_pretrained(..., fp32)) */
    snprintf(path, sizeof(path), "%s/vae", wan22_base);
    if (wan_vae_load(&p->vae, path)) goto fail;

    /* diff_dec: dual-expert Wan2.2 transformers */
    p->diff_dec.use_unipc = cfg_unipc;
    p->diff_dec.switch_dit_boundary = p->switch_dit_boundary;
    p->diff_dec.vae_scale_factor_temporal = 4;
    p->diff_dec.vae_scale_factor_spatial = 8;
    bn_sched_init(&p->diff_dec.scheduler,
                  cfg_unipc ? BN_SCHED_UNIPC : BN_SCHED_FLOW_MATCH,
                  1000, shift, 0.0 /* sigma_min */, 0 /* extra_one_step */);

    {
        bj_value *tcfg, *t2cfg;
        snprintf(path, sizeof(path), "%s/transformer/config.json", wan22_base);
        tcfg = bj_parse_file(path);
        snprintf(path, sizeof(path), "%s/transformer_2/config.json", wan22_base);
        t2cfg = bj_parse_file(path);
        if (!tcfg) {
            fprintf(stderr, "pipeline: missing transformer config under %s\n", wan22_base);
            bj_free(t2cfg);
            goto fail;
        }
        wan_transformer_config(&p->diff_dec.transformer, tcfg, cfg_srcid);
        if (t2cfg && !skip_t2) {
            wan_transformer_config(&p->diff_dec.transformer_2, t2cfg, cfg_srcid);
            p->diff_dec.has_transformer_2 = 1;
        }
        bj_free(tcfg);
        bj_free(t2cfg);
    }

    if (high_noise_ckpt && low_noise_ckpt) {
        /* bernini.weights.load_weights: prefix candidates + EMA preference */
        st_store *st = st_open_dir(high_noise_ckpt);
        if (!st) {
            fprintf(stderr, "pipeline: no safetensors in %s\n", high_noise_ckpt);
            goto fail;
        }
        int pi = st_pick_prefix(st, HIGH_NOISE_PREFIXES, 3);
        if (pi < 0 || wan_transformer_load(&p->diff_dec.transformer, st, HIGH_NOISE_PREFIXES[pi])) {
            st_close(st);
            goto fail;
        }
        st_close(st);
        if (p->diff_dec.has_transformer_2) {
            st = st_open_dir(low_noise_ckpt);
            if (!st) {
                fprintf(stderr, "pipeline: no safetensors in %s\n", low_noise_ckpt);
                goto fail;
            }
            pi = st_pick_prefix(st, LOW_NOISE_PREFIXES, 3);
            if (pi < 0 || wan_transformer_load(&p->diff_dec.transformer_2, st, LOW_NOISE_PREFIXES[pi])) {
                st_close(st);
                goto fail;
            }
            st_close(st);
        }
    } else {
        /* diffusers layout: transformer/ and transformer_2/ subfolders */
        snprintf(path, sizeof(path), "%s/transformer", wan22_base);
        st_store *st = st_open_dir(path);
        if (!st || wan_transformer_load(&p->diff_dec.transformer, st, "")) {
            if (st) st_close(st);
            fprintf(stderr, "pipeline: failed loading %s\n", path);
            goto fail;
        }
        st_close(st);
        if (p->diff_dec.has_transformer_2) {
            snprintf(path, sizeof(path), "%s/transformer_2", wan22_base);
            st = st_open_dir(path);
            if (!st || wan_transformer_load(&p->diff_dec.transformer_2, st, "")) {
                if (st) st_close(st);
                fprintf(stderr, "pipeline: failed loading %s\n", path);
                goto fail;
            }
            st_close(st);
        }
    }
    fprintf(stderr, "pipeline: loaded (use_unipc=%d shift=%.2f boundary=%.3f)\n",
            cfg_unipc, shift, p->switch_dit_boundary);
    return p;

fail:
    bernini_renderer_free(p);
    return NULL;
}

/* ------------------------------- generate ---------------------------------- */

void bernini_gen_params_init(bernini_gen_params_t *g) {
    memset(g, 0, sizeof(*g));
    g->neg_prompt = "";
    g->system_prompt = "";
    g->num_frames = 81;
    g->max_image_size = 624;
    g->height = 480;
    g->width = 832;
    g->num_inference_steps = 40;
    g->guidance_mode = "rv2v";
    g->omega_vid = 3.0;
    g->omega_img = 3.0;
    g->omega_txt = 4.0;
    g->omega_scale = 0.75;
    g->flow_shift = 5.0;
    g->seed = 42;
    g->fps = 16;
    g->eta = 0.5;
    g->norm_threshold[0] = 50.0;
    g->norm_threshold[1] = 50.0;
    g->momentum = -0.5;
    g->output_path = "output.mp4";
    g->write_output = 1;
}

static void mkdirs_for(const char *path) {
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s", path);
    for (char *q = buf + 1; *q; q++) {
        if (*q == '/') {
            *q = '\0';
            mkdir(buf, 0755);
            *q = '/';
        }
    }
}

int bernini_renderer_generate(bernini_renderer_pipeline_t *p,
                              const bernini_gen_params_t *g) {
    int rc = -1;
    char *prompt = prompt_clean(g->system_prompt, g->prompt);
    char *neg = prompt_clean("", g->neg_prompt ? g->neg_prompt : "");
    fprintf(stderr, "pipeline: prompt: %s\n", prompt);

    /* ---- encode visual conditions on the VAE ---- */
    int t = g->num_frames, h = -1, w = -1;
    bt_t *video_latents[16] = {0};
    int n_videos = 0;
    bt_t *image_latents[16] = {0};
    int n_images = 0;

    for (int i = 0; i < g->n_videos && i < 16; i++) {
        bt_t *pv = preprocess_video(g->videos[i], g->fps, g->max_image_size, g->num_frames);
        if (!pv) goto done;
        if (h < 0) {
            t = (int)pv->shape[1];
            h = (int)pv->shape[2];
            w = (int)pv->shape[3];
        }
        video_latents[n_videos++] = vae_encode_norm(p, pv);
        bt_free(pv);
    }
    if (g->image) {
        bt_t *pi = preprocess_image(g->image, g->max_image_size);
        if (!pi) goto done;
        if (h < 0) {
            h = (int)pi->shape[2];
            w = (int)pi->shape[3];
        }
        image_latents[n_images++] = vae_encode_norm(p, pi);
        bt_free(pi);
    }
    for (int i = 0; i < g->n_images && n_images < 16; i++) {
        bt_t *pi = preprocess_image(g->images[i], g->max_image_size);
        if (!pi) goto done;
        image_latents[n_images++] = vae_encode_norm(p, pi);
        bt_free(pi);
    }

    if (h < 0) {
        h = g->height;
        w = g->width;
    }
    h = bn_make_divisible(h, 16);
    w = bn_make_divisible(w, 16);

    /* ---- text encoding ---- */
    int lp, lu;
    float *prompt_embeds = encode_prompt(p, prompt, &lp);
    float *uncond_embeds = encode_prompt(p, neg, &lu);

    /* ---- diffusion sampling ---- */
    {
        gen_sample_args_t a;
        memset(&a, 0, sizeof(a));
        a.prompt_embeds = prompt_embeds;
        a.prompt_len = lp;
        a.uncond_embeds = uncond_embeds;
        a.uncond_len = lu;
        a.video_latents = video_latents;
        a.n_videos = n_videos;
        a.image_latents = image_latents;
        a.n_images = n_images;
        a.num_frames = t;
        a.width = w;
        a.height = h;
        a.num_inference_steps = g->num_inference_steps;
        a.guidance_mode = g->guidance_mode;
        a.omega_vid = g->omega_vid;
        a.omega_img = g->omega_img;
        a.omega_txt = g->omega_txt;
        a.omega_scale = g->omega_scale;
        a.flow_shift = g->flow_shift;
        a.seed = g->seed;
        a.eta = g->eta;
        a.norm_threshold[0] = g->norm_threshold[0];
        a.norm_threshold[1] = g->norm_threshold[1];
        a.momentum = g->momentum;

        bt_t *latents = gen_wanx22_sample(&p->diff_dec, &a);
        free(prompt_embeds);
        free(uncond_embeds);
        if (!latents) goto done;

        if (!g->write_output) {
            bt_free(latents);
            rc = 0;
            goto done;
        }

        /* ---- decode + save (mirrors _vae_decode + save_output) ---- */
        int z = (int)latents->shape[0];
        int64_t thw = latents->numel / z;
        for (int c = 0; c < z; c++) {
            float mu = p->vae.latents_mean[c], sd = p->vae.latents_std[c];
            float *ptr = latents->data + (int64_t)c * thw;
            for (int64_t i = 0; i < thw; i++) ptr[i] = ptr[i] * sd + mu;
        }
        bt_t *video = wan_vae_decode(&p->vae, latents);
        bt_free(latents);

        /* postprocess_video: (x/2 + 0.5).clamp(0,1), CHW->THWC */
        int T = (int)video->shape[1], H = (int)video->shape[2], W = (int)video->shape[3];
        float *frames = (float *)malloc(sizeof(float) * (size_t)T * H * W * 3);
        int64_t hw = (int64_t)H * W;
        for (int ti = 0; ti < T; ti++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < 3; c++) {
                        float v = video->data[((int64_t)c * T + ti) * hw + (int64_t)y * W + x];
                        v = v / 2.0f + 0.5f;
                        frames[(((int64_t)ti * H + y) * W + x) * 3 + c] =
                            v < 0 ? 0 : v > 1 ? 1 : v;
                    }
        bt_free(video);

        mkdirs_for(g->output_path);
        rc = bn_save_output(frames, T, H, W, g->output_path, g->fps);
        free(frames);
        if (rc == 0)
            fprintf(stderr, "pipeline: saved -> %s  (%d frames, %dx%d)\n",
                    g->output_path, T, h, w);
    }

done:
    for (int i = 0; i < n_videos; i++) bt_free(video_latents[i]);
    for (int i = 0; i < n_images; i++) bt_free(image_latents[i]);
    free(prompt);
    free(neg);
    return rc;
}

void bernini_renderer_free(bernini_renderer_pipeline_t *p) {
    if (!p) return;
    gen_wanx22_free(&p->diff_dec);
    wan_vae_free(&p->vae);
    umt5_free(&p->t5);
    bn_tokenizer_free(p->tokenizer);
    free(p);
}
