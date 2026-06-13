/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "data_utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int bn_make_divisible(int value, int stride) {
    int r = (int)llround((double)value / stride) * stride;
    return r < stride ? stride : r;
}

static void apply_scale(int w, int h, double scale, int stride, int *nw, int *nh) {
    *nw = bn_make_divisible((int)llround(w * scale), stride);
    *nh = bn_make_divisible((int)llround(h * scale), stride);
}

void bn_resize_target(int w, int h, int max_size, int min_size, int stride,
                      int *new_w, int *new_h) {
    int long_e = w > h ? w : h;
    int short_e = w < h ? w : h;
    double scale = (double)max_size / long_e;
    if (scale > 1.0) scale = 1.0;
    double min_scale = (double)min_size / short_e;
    if (scale < min_scale) scale = min_scale;
    apply_scale(w, h, scale, stride, new_w, new_h);
    int nl = *new_w > *new_h ? *new_w : *new_h;
    if (nl > max_size) {
        scale = (double)max_size / nl;
        apply_scale(*new_w, *new_h, scale, stride, new_w, new_h);
    }
}

/* Bicubic kernel (a = -0.75, as used by torchvision/OpenCV). */
static double cubic(double x, double a) {
    x = fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

/* Separable resize of one axis with antialias support scaling. */
static void resize_axis(const float *src, float *dst, int n_src, int n_dst,
                        int stride_src, int stride_dst, int n_lines,
                        int line_stride_src, int line_stride_dst) {
    double scale = (double)n_src / n_dst;
    double support = scale > 1.0 ? 2.0 * scale : 2.0;
    int ksize = (int)ceil(support) * 2 + 1;
    double inv = scale > 1.0 ? 1.0 / scale : 1.0;

    double *weights = (double *)malloc(sizeof(double) * (size_t)ksize);
    for (int o = 0; o < n_dst; o++) {
        double center = (o + 0.5) * scale - 0.5;
        int lo = (int)floor(center - support);
        int hi = (int)ceil(center + support);
        if (lo < 0) lo = 0;
        if (hi > n_src - 1) hi = n_src - 1;
        int kn = hi - lo + 1;
        double wsum = 0.0;
        for (int i = 0; i < kn; i++) {
            weights[i] = cubic((lo + i - center) * inv, -0.75);
            wsum += weights[i];
        }
        for (int l = 0; l < n_lines; l++) {
            const float *sline = src + (int64_t)l * line_stride_src;
            double acc = 0.0;
            for (int i = 0; i < kn; i++)
                acc += weights[i] * sline[(int64_t)(lo + i) * stride_src];
            dst[(int64_t)l * line_stride_dst + (int64_t)o * stride_dst] =
                (float)(wsum != 0.0 ? acc / wsum : 0.0);
        }
    }
    free(weights);
}

void bn_resize_bicubic_u8(const uint8_t *src, int sw, int sh,
                          uint8_t *dst, int dw, int dh, int ch) {
    /* resize in float (mirrors the uint8->float->resize->round path) */
    float *fsrc = (float *)malloc(sizeof(float) * (size_t)sw * sh * ch);
    for (int64_t i = 0; i < (int64_t)sw * sh * ch; i++) fsrc[i] = (float)src[i];
    /* horizontal: per (row, channel) line */
    float *mid = (float *)malloc(sizeof(float) * (size_t)dw * sh * ch);
    for (int c = 0; c < ch; c++)
        resize_axis(fsrc + c, mid + c, sw, dw, ch, ch, sh,
                    sw * ch, dw * ch);
    free(fsrc);
    /* vertical */
    float *out = (float *)malloc(sizeof(float) * (size_t)dw * dh * ch);
    for (int c = 0; c < ch; c++)
        for (int x = 0; x < dw; x++)
            resize_axis(mid + (int64_t)x * ch + c, out + (int64_t)x * ch + c,
                        sh, dh, dw * ch, dw * ch, 1, 0, 0);
    free(mid);
    for (int64_t i = 0; i < (int64_t)dw * dh * ch; i++) {
        float v = roundf(out[i]);
        dst[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    free(out);
}

bt_t *bn_vae_transform(const uint8_t *rgb, int w, int h,
                       int max_image_size, int min_image_size, int stride) {
    int nw, nh;
    bn_resize_target(w, h, max_image_size, min_image_size, stride, &nw, &nh);
    const uint8_t *use = rgb;
    uint8_t *resized = NULL;
    if (nw != w || nh != h) {
        resized = (uint8_t *)malloc((size_t)nw * nh * 3);
        bn_resize_bicubic_u8(rgb, w, h, resized, nw, nh, 3);
        use = resized;
    }
    /* ToTensor + Normalize(mean .5, std .5): CHW in [-1, 1] */
    bt_t *t = bt_new3(3, nh, nw);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++)
                t->data[((int64_t)c * nh + y) * nw + x] =
                    ((float)use[((int64_t)y * nw + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
    free(resized);
    return t;
}

int *bn_smart_video_nframes(int total_frames, double video_fps, double fps,
                            int frame_factor, int max_frames, int add_one, int *n_out) {
    double nf = (double)total_frames / video_fps * fps;
    long nframes;
    if (frame_factor > 0) {
        nframes = (long)floor(nf / frame_factor) * frame_factor + (add_one ? 1 : 0);
        long min_n = frame_factor + (add_one ? 1 : 0);
        if (nframes < min_n) nframes = min_n;
        if (video_fps == fps)
            total_frames = (int)(floor((double)total_frames / frame_factor) * frame_factor +
                                 (add_one ? 1 : 0));
    } else {
        nframes = (long)(nf + (add_one ? 1 : 0));
    }

    int count = (int)nframes;
    int *idx = (int *)malloc(sizeof(int) * (size_t)(count > 0 ? count : 1));
    for (int i = 0; i < count; i++) {
        double v = (count == 1) ? 0.0
                                : (double)i * (double)(total_frames - 1) / (double)(count - 1);
        idx[i] = (int)llround(v);
    }

    if (max_frames > 0) {
        long mf = max_frames;
        if (frame_factor > 0) mf = (long)floor((double)max_frames / frame_factor) * frame_factor;
        long limit = mf + (add_one ? 1 : 0);
        if (count > limit) count = (int)limit;
    }
    *n_out = count;
    return idx;
}
