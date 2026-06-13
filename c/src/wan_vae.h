/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* AutoencoderKLWan (replaces diffusers.AutoencoderKLWan for the pipeline's
 * encode/decode steps). Implements the Wan2.1-style architecture
 * (is_residual=False) with the chunked causal feature-cache semantics of the
 * reference: encode processes frames in 1+4k chunks, decode frame by frame,
 * so temporal up/downsampling treats the first frame as an image. */

#ifndef BERNINI_WAN_VAE_H
#define BERNINI_WAN_VAE_H

#include "bt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WAN_VAE_MAX_Z 64
#define WAN_VAE_MAX_CACHE 160

typedef struct { bt_t *w, *b; } vconv_t;

typedef struct {
    int in_dim, out_dim, has_shortcut;
    bt_t *norm1_g, *norm2_g;
    vconv_t conv1, conv2, shortcut;
} vres_t;

typedef struct {
    int dim;
    bt_t *norm_g;
    vconv_t qkv, proj;
} vattn_t;

/* mode: 0 none, 1 upsample2d, 2 upsample3d, 3 downsample2d, 4 downsample3d */
typedef struct {
    int mode, dim, out_dim;
    vconv_t conv;      /* spatial conv (resample.1) */
    vconv_t time_conv; /* causal (3,1,1) conv for the 3d modes */
} vresample_t;

typedef struct {
    int kind; /* 0 residual block, 1 resample */
    vres_t res;
    vresample_t rs;
} venc_layer_t;

typedef struct {
    vres_t res[8];
    int n_res;
    int has_upsampler;
    vresample_t up;
} vup_block_t;

typedef struct {
    int state; /* 0 none, 1 tensor, 2 "Rep" */
    bt_t *x;
} vcache_t;

typedef struct {
    /* config */
    int base_dim, z_dim, num_res_blocks;
    int dim_mult[4], n_mult;
    int temperal_downsample[4];
    float latents_mean[WAN_VAE_MAX_Z], latents_std[WAN_VAE_MAX_Z];

    /* encoder */
    vconv_t enc_conv_in;
    venc_layer_t enc_layers[32];
    int n_enc_layers;
    vres_t enc_mid_res[2];
    vattn_t enc_mid_attn;
    bt_t *enc_norm_out_g;
    vconv_t enc_conv_out;
    vconv_t quant_conv;

    /* decoder */
    vconv_t post_quant_conv;
    vconv_t dec_conv_in;
    vres_t dec_mid_res[2];
    vattn_t dec_mid_attn;
    vup_block_t up_blocks[4];
    int n_up_blocks;
    bt_t *dec_norm_out_g;
    vconv_t dec_conv_out;

    vcache_t cache[WAN_VAE_MAX_CACHE];
} wan_vae_t;

/* Load from a diffusers vae directory (config.json + *.safetensors). */
int  wan_vae_load(wan_vae_t *v, const char *dir);
void wan_vae_free(wan_vae_t *v);

/* pixels [3,T,H,W] in [-1,1] -> posterior mode (mean) [z, 1+(T-1)/4, H/8, W/8] */
bt_t *wan_vae_encode(wan_vae_t *v, const bt_t *pixels);
/* latents [z,T,H,W] -> pixels [3, 1+(T-1)*4, 8H, 8W] clamped to [-1,1] */
bt_t *wan_vae_decode(wan_vae_t *v, const bt_t *latents);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_WAN_VAE_H */
