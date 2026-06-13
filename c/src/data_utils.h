/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Video / image reading and VAE preprocessing (mirrors bernini/data_utils.py). */

#ifndef BERNINI_DATA_UTILS_H
#define BERNINI_DATA_UTILS_H

#include "bt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Round to the nearest multiple of `stride` (at least `stride`). */
int bn_make_divisible(int value, int stride);

/* MaxLongEdgeMinShortEdgeResize target-size logic: long edge <= max_size,
 * short edge >= min_size, snapped to stride. */
void bn_resize_target(int w, int h, int max_size, int min_size, int stride,
                      int *new_w, int *new_h);

/* Bicubic resize with antialias on uint8 RGB (HWC) -> uint8 RGB. */
void bn_resize_bicubic_u8(const uint8_t *src, int sw, int sh,
                          uint8_t *dst, int dw, int dh, int channels);

/* VAEVideoTransform: resize (max/min/stride) + normalize to [-1,1].
 * In: uint8 RGB HWC frame. Out: float CHW tensor (caller frees). */
bt_t *bn_vae_transform(const uint8_t *rgb, int w, int h,
                       int max_image_size, int min_image_size, int stride);

/* smart_video_nframes: pick frame indices so the sampled clip matches the
 * target fps / frame count. Returns malloc'd index array, count in *n_out. */
int *bn_smart_video_nframes(int total_frames, double video_fps, double fps,
                            int frame_factor, int max_frames, int add_one, int *n_out);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_DATA_UTILS_H */
