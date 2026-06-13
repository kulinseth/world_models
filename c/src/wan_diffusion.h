/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* GEN_Wanx22: Wan2.2 dual-expert (high-noise / low-noise) diffusion sampler
 * with APG / chained guidance (mirrors bernini/models/wan_diffusion.py). */

#ifndef BERNINI_WAN_DIFFUSION_H
#define BERNINI_WAN_DIFFUSION_H

#include "bt.h"
#include "scheduler.h"
#include "transformer_wan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    wan_transformer_t transformer;   /* high-noise expert */
    wan_transformer_t transformer_2; /* low-noise expert */
    int has_transformer_2;
    bn_scheduler scheduler;
    int    use_unipc;
    double switch_dit_boundary;
    int vae_scale_factor_temporal; /* 4 */
    int vae_scale_factor_spatial;  /* 8 */
} gen_wanx22_t;

typedef struct {
    /* text condition */
    const float *prompt_embeds;  /* [prompt_len, text_dim] */
    int64_t prompt_len;
    const float *uncond_embeds;  /* [uncond_len, text_dim] */
    int64_t uncond_len;
    /* visual conditions: normalized VAE latents, spatial [C,T,H,W] */
    bt_t **video_latents; int n_videos;
    bt_t **image_latents; int n_images; /* each [C,1,H,W] */
    /* target geometry / sampling params (same semantics as Python) */
    int num_frames, width, height;
    int num_inference_steps;
    const char *guidance_mode;
    double omega_vid, omega_img, omega_txt, omega_scale;
    double flow_shift;
    uint64_t seed;
    double eta;
    double norm_threshold[2];
    double momentum;
} gen_sample_args_t;

/* Run guided sampling; returns the predicted VAE latent [C,T,H,W] (caller
 * frees) and writes the dims. NULL on error (e.g. unknown guidance mode). */
bt_t *gen_wanx22_sample(gen_wanx22_t *g, const gen_sample_args_t *a);

void gen_wanx22_free(gen_wanx22_t *g);

/* packed <-> spatial rearranges (the einops _PACK/_UNPACK strings):
 * packed [(t h w), (ph pw c)] with h=H/2, w=W/2, ph=pw=2 */
void wan_pack_latent(const float *spatial, int c, int t, int h, int w, float *packed);
void wan_unpack_latent(const float *packed, int c, int t, int h, int w, float *spatial);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_WAN_DIFFUSION_H */
