/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Image / video IO (mirrors bernini/io_utils.py and the decord readers).
 * Decoding and H.264 encoding are delegated to the `ffmpeg` / `ffprobe`
 * CLIs via pipes; PPM is supported natively as a no-dependency fallback. */

#ifndef BERNINI_MEDIA_H
#define BERNINI_MEDIA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int      width, height;
    uint8_t *rgb; /* HWC, owned */
} bn_image_t;

typedef struct {
    int      width, height;
    int      n_frames;
    double   fps;
    uint8_t *rgb; /* [n_frames, H, W, 3], owned */
} bn_video_t;

/* Load any image ffmpeg can decode (PPM handled natively). 0 on success. */
int  bn_image_load(const char *path, bn_image_t *img);
void bn_image_free(bn_image_t *img);

/* Probe + decode a video to raw RGB frames. `indices` selects frames (may be
 * NULL to decode everything up to max_frames<=0 ? all : max). */
int  bn_video_load(const char *path, const int *indices, int n_indices, bn_video_t *vid);
int  bn_video_probe(const char *path, int *n_frames, double *fps, int *w, int *h);
void bn_video_free(bn_video_t *vid);

/* Save frames [T,H,W,3] float in [0,1]:
 * - T == 1 -> png (via ffmpeg; .ppm fallback)
 * - T > 1  -> H.264 mp4, libx264 + yuv420p, -crf 8 (mirrors io_utils)
 * Returns 0 on success. */
int bn_save_output(const float *frames, int t, int h, int w,
                   const char *path, int fps);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_MEDIA_H */
