/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "media.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ends_with(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && !strcasecmp(s + ls - lf, suf);
}

/* ------------------------------- PPM ------------------------------------- */

static int ppm_load(const char *path, bn_image_t *img) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[3] = {0};
    int w = 0, h = 0, maxv = 0;
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6")) { fclose(f); return -1; }
    int vals[3], got = 0;
    while (got < 3) {
        int c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); continue; }
        if (isspace(c)) continue;
        ungetc(c, f);
        if (fscanf(f, "%d", &vals[got]) != 1) { fclose(f); return -1; }
        got++;
    }
    fgetc(f); /* single whitespace after maxval */
    w = vals[0]; h = vals[1]; maxv = vals[2];
    (void)maxv;
    img->width = w;
    img->height = h;
    img->rgb = (uint8_t *)malloc((size_t)w * h * 3);
    if (fread(img->rgb, 1, (size_t)w * h * 3, f) != (size_t)w * h * 3) {
        fclose(f); free(img->rgb); return -1;
    }
    fclose(f);
    return 0;
}

static int ppm_save(const char *path, const uint8_t *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, (size_t)w * h * 3, f);
    fclose(f);
    return 0;
}

/* ------------------------------ ffprobe ----------------------------------- */

static int run_capture(const char *cmd, char *out, size_t out_sz) {
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    size_t n = fread(out, 1, out_sz - 1, p);
    out[n] = '\0';
    return pclose(p) == 0 ? 0 : -1;
}

static int probe_dims(const char *path, int *w, int *h) {
    char cmd[4608], buf[256];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams v:0 -show_entries stream=width,height "
             "-of csv=p=0 \"%s\" 2>/dev/null", path);
    if (run_capture(cmd, buf, sizeof(buf))) return -1;
    return sscanf(buf, "%d,%d", w, h) == 2 ? 0 : -1;
}

int bn_video_probe(const char *path, int *n_frames, double *fps, int *w, int *h) {
    if (probe_dims(path, w, h)) return -1;
    char cmd[4608], buf[256];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams v:0 -count_packets "
             "-show_entries stream=nb_read_packets,avg_frame_rate -of csv=p=0 \"%s\" 2>/dev/null",
             path);
    if (run_capture(cmd, buf, sizeof(buf))) return -1;
    int num = 0, den = 1;
    long frames = 0;
    if (sscanf(buf, "%d/%d,%ld", &num, &den, &frames) != 3) return -1;
    *fps = den > 0 ? (double)num / den : 0.0;
    *n_frames = (int)frames;
    return 0;
}

/* ------------------------------ image load -------------------------------- */

int bn_image_load(const char *path, bn_image_t *img) {
    if (ends_with(path, ".ppm")) return ppm_load(path, img);
    int w, h;
    if (probe_dims(path, &w, &h)) {
        fprintf(stderr, "media: ffprobe failed for %s (is ffmpeg installed?)\n", path);
        return -1;
    }
    char cmd[4608];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -v error -i \"%s\" -frames:v 1 -f rawvideo -pix_fmt rgb24 - 2>/dev/null",
             path);
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    size_t need = (size_t)w * h * 3;
    img->rgb = (uint8_t *)malloc(need);
    size_t got = fread(img->rgb, 1, need, p);
    pclose(p);
    if (got != need) {
        free(img->rgb);
        return -1;
    }
    img->width = w;
    img->height = h;
    return 0;
}

void bn_image_free(bn_image_t *img) {
    free(img->rgb);
    img->rgb = NULL;
}

/* ------------------------------ video load -------------------------------- */

int bn_video_load(const char *path, const int *indices, int n_indices, bn_video_t *vid) {
    int total, w, h;
    double fps;
    if (bn_video_probe(path, &total, &fps, &w, &h)) {
        fprintf(stderr, "media: cannot probe video %s\n", path);
        return -1;
    }
    /* decode every frame up to the max requested index, then pick */
    int max_idx = total - 1;
    if (indices) {
        max_idx = 0;
        for (int i = 0; i < n_indices; i++)
            if (indices[i] > max_idx) max_idx = indices[i];
        if (max_idx > total - 1) max_idx = total - 1;
    }
    char cmd[4608];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -v error -i \"%s\" -frames:v %d -f rawvideo -pix_fmt rgb24 - 2>/dev/null",
             path, max_idx + 1);
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    size_t fsz = (size_t)w * h * 3;
    uint8_t *all = (uint8_t *)malloc(fsz * (size_t)(max_idx + 1));
    int decoded = 0;
    while (decoded <= max_idx && fread(all + fsz * (size_t)decoded, 1, fsz, p) == fsz)
        decoded++;
    pclose(p);
    if (decoded == 0) {
        free(all);
        return -1;
    }

    int n = indices ? n_indices : decoded;
    vid->width = w;
    vid->height = h;
    vid->fps = fps;
    vid->n_frames = n;
    vid->rgb = (uint8_t *)malloc(fsz * (size_t)n);
    for (int i = 0; i < n; i++) {
        int src = indices ? indices[i] : i;
        if (src > decoded - 1) src = decoded - 1;
        if (src < 0) src = 0;
        memcpy(vid->rgb + fsz * (size_t)i, all + fsz * (size_t)src, fsz);
    }
    free(all);
    return 0;
}

void bn_video_free(bn_video_t *vid) {
    free(vid->rgb);
    vid->rgb = NULL;
}

/* -------------------------------- save ------------------------------------ */

static uint8_t *to_u8(const float *frames, int t, int h, int w) {
    size_t n = (size_t)t * h * w * 3;
    uint8_t *u8 = (uint8_t *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        float v = frames[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        u8[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    return u8;
}

int bn_save_output(const float *frames, int t, int h, int w, const char *path, int fps) {
    uint8_t *u8 = to_u8(frames, t, h, w);
    int rc = -1;
    if (t == 1) {
        if (ends_with(path, ".ppm")) {
            rc = ppm_save(path, u8, w, h);
        } else {
            char cmd[4608];
            snprintf(cmd, sizeof(cmd),
                     "ffmpeg -v error -y -f rawvideo -pix_fmt rgb24 -s %dx%d -i - "
                     "-frames:v 1 \"%s\" 2>/dev/null", w, h, path);
            FILE *p = popen(cmd, "w");
            if (p) {
                fwrite(u8, 1, (size_t)w * h * 3, p);
                rc = pclose(p) == 0 ? 0 : -1;
            }
        }
    } else {
        /* libx264 + yuv420p + crf 8, mirroring io_utils._imageio_mimwrite_h264 */
        char cmd[4608];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -v error -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - "
                 "-c:v libx264 -pix_fmt yuv420p -crf 8 \"%s\" 2>/dev/null",
                 w, h, fps, path);
        FILE *p = popen(cmd, "w");
        if (p) {
            fwrite(u8, 1, (size_t)t * h * w * 3, p);
            rc = pclose(p) == 0 ? 0 : -1;
        }
        if (rc != 0) {
            /* dependency-free fallback: numbered PPM frames */
            fprintf(stderr, "media: ffmpeg unavailable, writing PPM frames next to %s\n", path);
            char fp[4360];
            rc = 0;
            for (int i = 0; i < t; i++) {
                snprintf(fp, sizeof(fp), "%s.%05d.ppm", path, i);
                rc |= ppm_save(fp, u8 + (size_t)i * h * w * 3, w, h);
            }
        }
    }
    free(u8);
    return rc;
}
