/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Self-contained unit tests for the numerics-critical pieces of the C port.
 * Reference values mirror the Python implementations. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bjson.h"
#include "bt.h"
#include "data_utils.h"
#include "guidance.h"
#include "rng.h"
#include "scheduler.h"
#include "attention.h"
#include "wan_diffusion.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(fabs((a) - (b)) <= (tol), "%g != %g", (double)(a), (double)(b))

static void test_flow_match_sigmas(void) {
    /* scheduler.py: linspace(1, 0, 4) shifted by 5: s' = 5s / (1 + 4s) */
    bn_scheduler s;
    bn_sched_init(&s, BN_SCHED_FLOW_MATCH, 1000, 3.0, 0.0, 0);
    bn_sched_set_timesteps(&s, 4, 5.0);
    double expect[4] = {1.0, 5.0 * (2.0 / 3) / (1 + 4.0 * 2 / 3),
                        5.0 * (1.0 / 3) / (1 + 4.0 / 3), 0.0};
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(s.sigmas[i], expect[i], 1e-9);
        CHECK_NEAR(s.timesteps[i], expect[i] * 1000.0, 1e-6);
    }
    /* one Euler step: x += v * (sigma1 - sigma0) */
    float x = 2.0f, v = 1.0f;
    bn_sched_step(&s, &v, s.timesteps[0], &x, 1);
    CHECK_NEAR(x, 2.0 + (expect[1] - expect[0]), 1e-6);
    bn_sched_free(&s);
}

static void test_unipc_converges(void) {
    /* flow ODE with ground-truth x0 = 0: v = x_t / sigma_t. UniPC must drive
     * the sample to ~0 over the schedule. */
    bn_scheduler s;
    bn_sched_init(&s, BN_SCHED_UNIPC, 1000, 5.0, 0.0, 0);
    bn_sched_set_timesteps(&s, 20, 0);
    CHECK(s.num_steps == 20, "unipc steps");
    CHECK_NEAR(s.sigmas[20], 0.0, 1e-12); /* terminal zero sigma */
    CHECK(fabs(s.sigmas[0] - 1.0) > 1e-9, "first sigma nudged off 1.0");
    float x = 3.0f;
    for (int i = 0; i < s.num_steps; i++) {
        double sigma = s.sigmas[i];
        float v = (float)((double)x / sigma);
        bn_sched_step(&s, &v, s.timesteps[i], &x, 1);
    }
    CHECK(fabs(x) < 1e-3, "unipc did not converge: %g", (double)x);
    bn_sched_free(&s);
}

static void test_pack_roundtrip(void) {
    int C = 3, T = 2, H = 4, W = 6;
    int64_t n = (int64_t)C * T * H * W;
    float *sp = (float *)malloc(sizeof(float) * n);
    for (int64_t i = 0; i < n; i++) sp[i] = (float)i * 0.5f;
    float *packed = (float *)malloc(sizeof(float) * n);
    float *back = (float *)malloc(sizeof(float) * n);
    wan_pack_latent(sp, C, T, H, W, packed);
    wan_unpack_latent(packed, C, T, H, W, back);
    for (int64_t i = 0; i < n; i++) CHECK(sp[i] == back[i], "pack roundtrip @%lld", (long long)i);
    /* spot-check the einops layout: token (t,y,x), channel (ph pw c) */
    /* packed[token=0, ph=0 pw=1 c=2] == sp[c=2, t=0, y=0, x=1] */
    CHECK(packed[0 * (4 * C) + (0 * 2 + 1) * C + 2] == sp[(int64_t)2 * T * H * W + 0 + 1],
          "einops channel order");
    free(sp); free(packed); free(back);
}

static void test_norms(void) {
    /* layernorm against hand-computed values */
    float x[4] = {1, 2, 3, 4};
    bt_layernorm(x, 1, 4, NULL, NULL, 0.0f);
    double std = sqrt(1.25);
    CHECK_NEAR(x[0], (1 - 2.5) / std, 1e-5);
    CHECK_NEAR(x[3], (4 - 2.5) / std, 1e-5);

    float y[2] = {3, 4};
    bt_rmsnorm(y, 1, 2, NULL, 0.0f);
    double rms = sqrt((9.0 + 16.0) / 2.0);
    CHECK_NEAR(y[0], 3.0 / rms, 1e-5);

    float z[2] = {3, 4};
    bt_l2norm_rms(z, 1, 2, NULL);
    CHECK_NEAR(z[0], 3.0 / 5.0 * sqrt(2.0), 1e-5);
}

static void test_attention(void) {
    /* single segment, 1 head, head_dim 2: verify against naive softmax */
    float q[4] = {1, 0, 0, 1};  /* 2 tokens */
    float k[4] = {1, 0, 0, 1};
    float v[4] = {1, 2, 3, 4};
    float out[4];
    int64_t cu[2] = {0, 2};
    bn_varlen_attention(q, k, v, out, cu, cu, 1, 1, 2, 0);
    double s = 1.0 / sqrt(2.0);
    double w00 = exp(s), w01 = exp(0.0);
    double denom = w00 + w01;
    CHECK_NEAR(out[0], (w00 * 1 + w01 * 3) / denom, 1e-5);
    CHECK_NEAR(out[1], (w00 * 2 + w01 * 4) / denom, 1e-5);
}

static void test_apg(void) {
    /* apg_delta: delta parallel to ref -> scaled by parallel_scale */
    float delta[2] = {2, 0}, ref[2] = {1, 0};
    bn_apg_delta(delta, ref, 2, 0.2, 1.0);
    CHECK_NEAR(delta[0], 0.4, 1e-6);
    CHECK_NEAR(delta[1], 0.0, 1e-6);
    /* orthogonal direction passes through */
    float d2[2] = {0, 2}, r2[2] = {1, 0};
    bn_apg_delta(d2, r2, 2, 0.2, 1.0);
    CHECK_NEAR(d2[1], 2.0, 1e-6);
}

static void test_json(void) {
    bj_value *v = bj_parse("{\"a\": 1.5, \"b\": [1, 2, 3], \"c\": \"x\\ny\", \"d\": true, \"e\": null}");
    CHECK(v != NULL, "parse failed");
    CHECK_NEAR(bj_get_num(v, "a", 0), 1.5, 0);
    float arr[3];
    CHECK(bj_get_farray(v, "b", arr, 3) == 3, "array len");
    CHECK_NEAR(arr[2], 3.0, 0);
    CHECK(!strcmp(bj_get_str(v, "c", ""), "x\ny"), "string escape");
    CHECK(bj_get_bool(v, "d", 0) == 1, "bool");
    bj_free(v);
}

static void test_data_utils(void) {
    CHECK(bn_make_divisible(481, 16) == 480, "make_divisible 481");
    CHECK(bn_make_divisible(7, 16) == 16, "make_divisible floor");
    int nw, nh;
    /* 1920x1080 with max 624 stride 16: scale .325 -> 624x352 */
    bn_resize_target(1920, 1080, 624, 1, 16, &nw, &nh);
    CHECK(nw == 624 && nh == 352, "resize target %dx%d", nw, nh);
    int n;
    int *idx = bn_smart_video_nframes(100, 25.0, 16.0, 4, 81, 1, &n);
    CHECK(n == 65, "smart nframes %d", n); /* 100/25*16=64 -> 64+1 */
    CHECK(idx[0] == 0 && idx[n - 1] == 99, "frame index range");
    free(idx);
    /* identity resize is a no-op path upstream; check interpolation midpoint */
    uint8_t src[2 * 1] = {0, 100};
    uint8_t dst[2];
    bn_resize_bicubic_u8(src, 2, 1, dst, 2, 1, 1);
    CHECK(dst[0] == 0 && dst[1] == 100, "resize identity %d %d", dst[0], dst[1]);
}

static void test_rng(void) {
    bn_rng r;
    bn_rng_seed(&r, 42);
    double mean = 0, var = 0;
    int n = 20000;
    float *buf = (float *)malloc(sizeof(float) * n);
    bn_rng_randn(&r, buf, n);
    for (int i = 0; i < n; i++) mean += buf[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (buf[i] - mean) * (buf[i] - mean);
    var /= n;
    CHECK(fabs(mean) < 0.05, "randn mean %g", mean);
    CHECK(fabs(var - 1.0) < 0.05, "randn var %g", var);
    /* determinism */
    bn_rng r2;
    bn_rng_seed(&r2, 42);
    float first;
    bn_rng_randn(&r2, &first, 1);
    CHECK(first == buf[0], "rng determinism");
    free(buf);
}

int main(void) {
    test_flow_match_sigmas();
    test_unipc_converges();
    test_pack_roundtrip();
    test_norms();
    test_attention();
    test_apg();
    test_json();
    test_data_utils();
    test_rng();
    if (failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", failures);
    return 1;
}
