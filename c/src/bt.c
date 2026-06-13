/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "bt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef BT_USE_ACCELERATE
/* Declare the one cblas entry point we use instead of pulling in the
 * Accelerate umbrella header (its Sparse headers do not compile under
 * strict -std=c11 on recent SDKs). Linked via -framework Accelerate. */
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
extern void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa,
                        enum CBLAS_TRANSPOSE transb, int m, int n, int k,
                        float alpha, const float *a, int lda, const float *b,
                        int ldb, float beta, float *c, int ldc);
#endif

/* ---- lifecycle ---- */

bt_t *bt_new(int ndim, const int64_t *shape) {
    bt_t *t = (bt_t *)calloc(1, sizeof(bt_t));
    t->ndim = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    t->data = (float *)malloc(sizeof(float) * (size_t)t->numel);
    if (!t->data) {
        fprintf(stderr, "bt: out of memory allocating %lld floats\n", (long long)t->numel);
        exit(1);
    }
    return t;
}

bt_t *bt_new1(int64_t d0) { int64_t s[1] = {d0}; return bt_new(1, s); }
bt_t *bt_new2(int64_t d0, int64_t d1) { int64_t s[2] = {d0, d1}; return bt_new(2, s); }
bt_t *bt_new3(int64_t d0, int64_t d1, int64_t d2) { int64_t s[3] = {d0, d1, d2}; return bt_new(3, s); }
bt_t *bt_new4(int64_t d0, int64_t d1, int64_t d2, int64_t d3) { int64_t s[4] = {d0, d1, d2, d3}; return bt_new(4, s); }
bt_t *bt_new5(int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    int64_t s[5] = {d0, d1, d2, d3, d4}; return bt_new(5, s);
}

bt_t *bt_zeros(int ndim, const int64_t *shape) {
    bt_t *t = bt_new(ndim, shape);
    memset(t->data, 0, sizeof(float) * (size_t)t->numel);
    return t;
}

bt_t *bt_clone(const bt_t *src) {
    bt_t *t = bt_new(src->ndim, src->shape);
    memcpy(t->data, src->data, sizeof(float) * (size_t)src->numel);
    return t;
}

void bt_free(bt_t *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

void bt_fill(bt_t *t, float v) {
    for (int64_t i = 0; i < t->numel; i++) t->data[i] = v;
}

/* ---- elementwise ---- */

void bt_add(float *a, const float *b, int64_t n) { for (int64_t i = 0; i < n; i++) a[i] += b[i]; }
void bt_sub(float *a, const float *b, int64_t n) { for (int64_t i = 0; i < n; i++) a[i] -= b[i]; }
void bt_mul(float *a, const float *b, int64_t n) { for (int64_t i = 0; i < n; i++) a[i] *= b[i]; }
void bt_scale(float *a, float s, int64_t n) { for (int64_t i = 0; i < n; i++) a[i] *= s; }
void bt_axpy(float *y, float a, const float *x, int64_t n) { for (int64_t i = 0; i < n; i++) y[i] += a * x[i]; }
void bt_copy(float *dst, const float *src, int64_t n) { memcpy(dst, src, sizeof(float) * (size_t)n); }

void bt_clamp(float *a, float lo, float hi, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        if (a[i] < lo) a[i] = lo;
        else if (a[i] > hi) a[i] = hi;
    }
}

/* ---- activations ---- */

void bt_silu(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void bt_gelu_tanh(float *x, int64_t n) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

void bt_gelu_erf(float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.7071067811865476f));
}

/* ---- normalization ---- */

void bt_layernorm(float *x, int64_t rows, int64_t dim,
                  const float *gamma, const float *beta, float eps) {
#pragma omp parallel for
    for (int64_t r = 0; r < rows; r++) {
        float *row = x + r * dim;
        double mean = 0.0, var = 0.0;
        for (int64_t i = 0; i < dim; i++) mean += row[i];
        mean /= (double)dim;
        for (int64_t i = 0; i < dim; i++) {
            double d = row[i] - mean;
            var += d * d;
        }
        var /= (double)dim;
        float inv = (float)(1.0 / sqrt(var + (double)eps));
        for (int64_t i = 0; i < dim; i++) {
            float v = ((float)(row[i] - mean)) * inv;
            if (gamma) v *= gamma[i];
            if (beta) v += beta[i];
            row[i] = v;
        }
    }
}

void bt_rmsnorm(float *x, int64_t rows, int64_t dim, const float *gamma, float eps) {
#pragma omp parallel for
    for (int64_t r = 0; r < rows; r++) {
        float *row = x + r * dim;
        double ss = 0.0;
        for (int64_t i = 0; i < dim; i++) ss += (double)row[i] * (double)row[i];
        float inv = (float)(1.0 / sqrt(ss / (double)dim + (double)eps));
        for (int64_t i = 0; i < dim; i++) {
            float v = row[i] * inv;
            if (gamma) v *= gamma[i];
            row[i] = v;
        }
    }
}

void bt_l2norm_rms(float *x, int64_t rows, int64_t dim, const float *gamma) {
    const double scale = sqrt((double)dim);
#pragma omp parallel for
    for (int64_t r = 0; r < rows; r++) {
        float *row = x + r * dim;
        double ss = 0.0;
        for (int64_t i = 0; i < dim; i++) ss += (double)row[i] * (double)row[i];
        double nrm = sqrt(ss);
        if (nrm < 1e-12) nrm = 1e-12;
        float inv = (float)(scale / nrm);
        for (int64_t i = 0; i < dim; i++) {
            float v = row[i] * inv;
            if (gamma) v *= gamma[i];
            row[i] = v;
        }
    }
}

void bt_softmax_rows(float *x, int64_t rows, int64_t dim) {
#pragma omp parallel for
    for (int64_t r = 0; r < rows; r++) {
        float *row = x + r * dim;
        float mx = row[0];
        for (int64_t i = 1; i < dim; i++)
            if (row[i] > mx) mx = row[i];
        double sum = 0.0;
        for (int64_t i = 0; i < dim; i++) {
            row[i] = expf(row[i] - mx);
            sum += row[i];
        }
        float inv = (float)(1.0 / sum);
        for (int64_t i = 0; i < dim; i++) row[i] *= inv;
    }
}

/* ---- matmul / linear ---- */

void bt_matmul(const float *a, const float *b, float *c,
               int64_t m, int64_t k, int64_t n, int accumulate) {
#ifdef BT_USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k, 1.0f, a, (int)k, b, (int)n,
                accumulate ? 1.0f : 0.0f, c, (int)n);
#else
    if (!accumulate) memset(c, 0, sizeof(float) * (size_t)(m * n));
#pragma omp parallel for
    for (int64_t i = 0; i < m; i++) {
        const float *arow = a + i * k;
        float *crow = c + i * n;
        for (int64_t p = 0; p < k; p++) {
            float av = arow[p];
            if (av == 0.0f) continue;
            const float *brow = b + p * n;
            for (int64_t j = 0; j < n; j++) crow[j] += av * brow[j];
        }
    }
#endif
}

void bt_linear(const float *x, const float *w, const float *b,
               float *y, int64_t m, int64_t k, int64_t n) {
#ifdef BT_USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)n, (int)k, 1.0f, x, (int)k, w, (int)k, 0.0f, y, (int)n);
    if (b) {
#pragma omp parallel for
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) y[i * n + j] += b[j];
    }
#else
#pragma omp parallel for
    for (int64_t i = 0; i < m; i++) {
        const float *xrow = x + i * k;
        float *yrow = y + i * n;
        for (int64_t j = 0; j < n; j++) {
            const float *wrow = w + j * k;
            double acc = b ? (double)b[j] : 0.0;
            for (int64_t p = 0; p < k; p++) acc += (double)xrow[p] * (double)wrow[p];
            yrow[j] = (float)acc;
        }
    }
#endif
}

/* ---- convolutions ---- */

void bt_conv3d(const float *in, int cin, int t, int h, int w,
               const float *wt, const float *bias, int cout,
               int kt, int kh, int kw, int st, int sh, int sw,
               int pt_front, int pt_back, int ph, int pw,
               float *out, int *ot_, int *oh_, int *ow_) {
    int ot = (t + pt_front + pt_back - kt) / st + 1;
    int oh = (h + 2 * ph - kh) / sh + 1;
    int ow = (w + 2 * pw - kw) / sw + 1;
    *ot_ = ot; *oh_ = oh; *ow_ = ow;

    const int64_t in_thw = (int64_t)t * h * w;
    const int64_t in_hw = (int64_t)h * w;
    const int64_t out_thw = (int64_t)ot * oh * ow;
    const int64_t w_cin = (int64_t)kt * kh * kw;

#pragma omp parallel for collapse(2)
    for (int oc = 0; oc < cout; oc++) {
        for (int z = 0; z < ot; z++) {
            float *optr = out + oc * out_thw + (int64_t)z * oh * ow;
            const float *wbase = wt + (int64_t)oc * cin * w_cin;
            int it0 = z * st - pt_front;
            for (int y = 0; y < oh; y++) {
                int iy0 = y * sh - ph;
                for (int x2 = 0; x2 < ow; x2++) {
                    int ix0 = x2 * sw - pw;
                    double acc = bias ? (double)bias[oc] : 0.0;
                    for (int ic = 0; ic < cin; ic++) {
                        const float *ibase = in + (int64_t)ic * in_thw;
                        const float *wc = wbase + (int64_t)ic * w_cin;
                        for (int dz = 0; dz < kt; dz++) {
                            int iz = it0 + dz;
                            if (iz < 0 || iz >= t) continue;
                            for (int dy = 0; dy < kh; dy++) {
                                int iy = iy0 + dy;
                                if (iy < 0 || iy >= h) continue;
                                const float *irow = ibase + (int64_t)iz * in_hw + (int64_t)iy * w;
                                const float *wrow = wc + ((int64_t)dz * kh + dy) * kw;
                                for (int dx = 0; dx < kw; dx++) {
                                    int ix = ix0 + dx;
                                    if (ix < 0 || ix >= w) continue;
                                    acc += (double)irow[ix] * (double)wrow[dx];
                                }
                            }
                        }
                    }
                    optr[(int64_t)y * ow + x2] = (float)acc;
                }
            }
        }
    }
}

void bt_conv2d(const float *in, int cin, int h, int w,
               const float *wt, const float *bias, int cout,
               int kh, int kw, int sh, int sw,
               int pl, int pr, int pt, int pb,
               float *out, int *oh_, int *ow_) {
    int oh = (h + pt + pb - kh) / sh + 1;
    int ow = (w + pl + pr - kw) / sw + 1;
    *oh_ = oh; *ow_ = ow;
    const int64_t in_hw = (int64_t)h * w;
    const int64_t out_hw = (int64_t)oh * ow;

#pragma omp parallel for
    for (int oc = 0; oc < cout; oc++) {
        float *optr = out + (int64_t)oc * out_hw;
        const float *wbase = wt + (int64_t)oc * cin * kh * kw;
        for (int y = 0; y < oh; y++) {
            int iy0 = y * sh - pt;
            for (int x2 = 0; x2 < ow; x2++) {
                int ix0 = x2 * sw - pl;
                double acc = bias ? (double)bias[oc] : 0.0;
                for (int ic = 0; ic < cin; ic++) {
                    const float *ibase = in + (int64_t)ic * in_hw;
                    const float *wc = wbase + (int64_t)ic * kh * kw;
                    for (int dy = 0; dy < kh; dy++) {
                        int iy = iy0 + dy;
                        if (iy < 0 || iy >= h) continue;
                        const float *irow = ibase + (int64_t)iy * w;
                        const float *wrow = wc + (int64_t)dy * kw;
                        for (int dx = 0; dx < kw; dx++) {
                            int ix = ix0 + dx;
                            if (ix < 0 || ix >= w) continue;
                            acc += (double)irow[ix] * (double)wrow[dx];
                        }
                    }
                }
                optr[(int64_t)y * ow + x2] = (float)acc;
            }
        }
    }
}

void bt_upsample_nearest2x(const float *in, int c, int h, int w, float *out) {
    const int oh = 2 * h, ow = 2 * w;
#pragma omp parallel for
    for (int ch = 0; ch < c; ch++) {
        const float *ip = in + (int64_t)ch * h * w;
        float *op = out + (int64_t)ch * oh * ow;
        for (int y = 0; y < oh; y++) {
            const float *irow = ip + (int64_t)(y / 2) * w;
            float *orow = op + (int64_t)y * ow;
            for (int x = 0; x < ow; x++) orow[x] = irow[x / 2];
        }
    }
}

/* ---- reductions ---- */

double bt_dot(const float *a, const float *b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}

double bt_sumsq(const float *a, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)a[i] * (double)a[i];
    return s;
}
