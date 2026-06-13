/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

/* bt: minimal float32 tensor core for the Bernini C inference port.
 *
 * All tensors are contiguous row-major float32 with batch size 1 assumed by
 * the model code. This replaces the subset of torch ops the Python pipeline
 * uses (linear, layernorm/rmsnorm, conv2d/conv3d, softmax, activations). */

#ifndef BERNINI_BT_H
#define BERNINI_BT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BT_MAX_DIMS 6

typedef struct {
    int     ndim;
    int64_t shape[BT_MAX_DIMS];
    int64_t numel;
    float  *data;
} bt_t;

/* ---- lifecycle ---- */
bt_t *bt_new(int ndim, const int64_t *shape);
bt_t *bt_new1(int64_t d0);
bt_t *bt_new2(int64_t d0, int64_t d1);
bt_t *bt_new3(int64_t d0, int64_t d1, int64_t d2);
bt_t *bt_new4(int64_t d0, int64_t d1, int64_t d2, int64_t d3);
bt_t *bt_new5(int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4);
bt_t *bt_zeros(int ndim, const int64_t *shape);
bt_t *bt_clone(const bt_t *t);
void  bt_free(bt_t *t);
void  bt_fill(bt_t *t, float v);

/* ---- elementwise (raw pointers, length n) ---- */
void bt_add(float *a, const float *b, int64_t n);           /* a += b        */
void bt_sub(float *a, const float *b, int64_t n);           /* a -= b        */
void bt_mul(float *a, const float *b, int64_t n);           /* a *= b        */
void bt_scale(float *a, float s, int64_t n);                /* a *= s        */
void bt_axpy(float *y, float a, const float *x, int64_t n); /* y += a*x      */
void bt_copy(float *dst, const float *src, int64_t n);
void bt_clamp(float *a, float lo, float hi, int64_t n);

/* ---- activations (in place) ---- */
void bt_silu(float *x, int64_t n);
void bt_gelu_tanh(float *x, int64_t n); /* gelu(approximate="tanh") / gelu_new */
void bt_gelu_erf(float *x, int64_t n);  /* exact gelu */

/* ---- normalization (in place over rows of length dim) ---- */
void bt_layernorm(float *x, int64_t rows, int64_t dim,
                  const float *gamma, const float *beta, float eps);
void bt_rmsnorm(float *x, int64_t rows, int64_t dim, const float *gamma, float eps);
/* F.normalize-style RMS used by the Wan VAE: x / max(||x||, 1e-12) * sqrt(dim) * gamma */
void bt_l2norm_rms(float *x, int64_t rows, int64_t dim, const float *gamma);

void bt_softmax_rows(float *x, int64_t rows, int64_t dim);

/* ---- matmul / linear ----
 * bt_linear computes y = x @ W^T + b with the torch nn.Linear layout:
 * x [m,k], W [n,k] (row-major, as stored in checkpoints), b [n] or NULL. */
void bt_linear(const float *x, const float *w, const float *b,
               float *y, int64_t m, int64_t k, int64_t n);
/* c = a @ b with a [m,k], b [k,n]; if accumulate, c += a @ b */
void bt_matmul(const float *a, const float *b, float *c,
               int64_t m, int64_t k, int64_t n, int accumulate);

/* ---- convolutions (batch 1) ----
 * in  [Cin, T, H, W], w [Cout, Cin, kt, kh, kw], b [Cout] or NULL.
 * Time padding is asymmetric (pt_front zeros before, pt_back after) to
 * support the causal convs of the Wan VAE; spatial padding is symmetric.
 * out must hold Cout*oT*oH*oW floats; output dims are returned. */
void bt_conv3d(const float *in, int cin, int t, int h, int w,
               const float *wt, const float *bias, int cout,
               int kt, int kh, int kw, int st, int sh, int sw,
               int pt_front, int pt_back, int ph, int pw,
               float *out, int *ot, int *oh, int *ow);
/* conv2d on [Cin,H,W]; asymmetric spatial padding (left/right/top/bottom)
 * to support the ZeroPad2d((0,1,0,1)) used by Wan VAE downsampling. */
void bt_conv2d(const float *in, int cin, int h, int w,
               const float *wt, const float *bias, int cout,
               int kh, int kw, int sh, int sw,
               int pl, int pr, int pt, int pb,
               float *out, int *oh, int *ow);

/* nearest-exact 2x spatial upsample: [C,H,W] -> [C,2H,2W] */
void bt_upsample_nearest2x(const float *in, int c, int h, int w, float *out);

/* ---- reductions ---- */
double bt_dot(const float *a, const float *b, int64_t n);
double bt_sumsq(const float *a, int64_t n);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_BT_H */
