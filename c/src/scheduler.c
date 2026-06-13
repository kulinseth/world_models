/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "scheduler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void free_state(bn_scheduler *s) {
    free(s->model_outputs[0]);
    free(s->model_outputs[1]);
    free(s->last_sample);
    s->model_outputs[0] = s->model_outputs[1] = NULL;
    s->last_sample = NULL;
    s->state_numel = 0;
}

void bn_sched_init(bn_scheduler *s, bn_sched_type type, int num_train_timesteps,
                   double shift, double sigma_min, int extra_one_step) {
    memset(s, 0, sizeof(*s));
    s->type = type;
    s->num_train_timesteps = num_train_timesteps;
    s->shift = shift;
    s->sigma_max = 1.0;
    s->sigma_min = sigma_min;
    s->extra_one_step = extra_one_step;
    s->solver_order = 2;
    s->step_index = -1;
}

void bn_sched_set_timesteps(bn_scheduler *s, int n, double shift_override) {
    if (shift_override > 0) s->shift = shift_override;
    free(s->sigmas);
    free(s->timesteps);
    free_state(s);

    if (s->type == BN_SCHED_FLOW_MATCH) {
        /* scheduler.py: linspace(sigma_start, sigma_min, n[+1])[:n], shifted */
        s->num_steps = n;
        s->sigmas = (double *)malloc(sizeof(double) * (size_t)n);
        s->timesteps = (double *)malloc(sizeof(double) * (size_t)n);
        int pts = s->extra_one_step ? n + 1 : n;
        for (int i = 0; i < n; i++) {
            double sig = s->sigma_max +
                         (s->sigma_min - s->sigma_max) * ((pts == 1) ? 0.0 : (double)i / (double)(pts - 1));
            sig = s->shift * sig / (1.0 + (s->shift - 1.0) * sig);
            s->sigmas[i] = sig;
            s->timesteps[i] = sig * s->num_train_timesteps;
        }
    } else {
        /* UniPC flow sigmas: linspace(1, 1/num_train, n+1)[:-1], shifted,
         * first sigma nudged off 1.0, terminal sigma 0 appended. */
        s->num_steps = n;
        s->sigmas = (double *)malloc(sizeof(double) * (size_t)(n + 1));
        s->timesteps = (double *)malloc(sizeof(double) * (size_t)n);
        for (int i = 0; i < n; i++) {
            double sig = 1.0 + ((1.0 / s->num_train_timesteps) - 1.0) * ((double)i / (double)n);
            sig = s->shift * sig / (1.0 + (s->shift - 1.0) * sig);
            s->sigmas[i] = sig;
        }
        if (fabs(s->sigmas[0] - 1.0) < 1e-6) s->sigmas[0] -= 1e-6;
        s->sigmas[n] = 0.0;
        for (int i = 0; i < n; i++) s->timesteps[i] = s->sigmas[i] * s->num_train_timesteps;
    }
    s->step_index = -1;
    s->lower_order_nums = 0;
    s->this_order = 1;
}

static int nearest_timestep_index(const bn_scheduler *s, double t) {
    int best = 0;
    double bd = fabs(s->timesteps[0] - t);
    for (int i = 1; i < s->num_steps; i++) {
        double d = fabs(s->timesteps[i] - t);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

/* ---------------- FlowMatch step (scheduler.py FlowMatchScheduler.step) --- */

static void flow_match_step(bn_scheduler *s, const float *v, double t, float *x, int64_t n) {
    int id = nearest_timestep_index(s, t);
    double sigma = s->sigmas[id];
    double sigma_next = (id + 1 < s->num_steps) ? s->sigmas[id + 1] : 0.0;
    float d = (float)(sigma_next - sigma);
    for (int64_t i = 0; i < n; i++) x[i] += v[i] * d;
}

/* ---------------- UniPC-bh2 (flow, predict_x0, order<=2) ------------------ */

static double lambda_of(double sigma) {
    /* flow sigmas: alpha_t = 1 - sigma, sigma_t = sigma */
    return log(1.0 - sigma) - log(sigma);
}

/* predictor: from `sample` at step_index to step_index+1 */
static void unipc_p_update(bn_scheduler *s, const float *sample, float *out, int64_t n, int order) {
    double sigma_t = s->sigmas[s->step_index + 1];
    double sigma_s0 = s->sigmas[s->step_index];
    const float *m0 = s->model_outputs[1];

    if (sigma_t < 1e-12) { /* terminal step: limit of the update is x0 */
        memcpy(out, m0, sizeof(float) * (size_t)n);
        return;
    }
    double alpha_t = 1.0 - sigma_t;
    double h = lambda_of(sigma_t) - lambda_of(sigma_s0);
    double hh = -h;
    double h_phi_1 = expm1(hh);
    double b_h = expm1(hh); /* bh2 */

    double c_x = sigma_t / sigma_s0;
    double c_m0 = -alpha_t * h_phi_1;

    if (order == 2) {
        double sigma_s1 = s->sigmas[s->step_index - 1];
        double r0 = (lambda_of(sigma_s1) - lambda_of(sigma_s0)) / h;
        const float *m1 = s->model_outputs[0];
        double rho = 0.5; /* simplified order-2 predictor */
        double c_d1 = -alpha_t * b_h * rho / r0;
        for (int64_t i = 0; i < n; i++)
            out[i] = (float)(c_x * sample[i] + c_m0 * m0[i] + c_d1 * (m1[i] - m0[i]));
    } else {
        for (int64_t i = 0; i < n; i++)
            out[i] = (float)(c_x * sample[i] + c_m0 * m0[i]);
    }
}

/* corrector: refine `this_sample` (at step_index) from `last_sample` */
static void unipc_c_update(bn_scheduler *s, const float *model_t, float *this_sample,
                           int64_t n, int order) {
    double sigma_t = s->sigmas[s->step_index];
    double sigma_s0 = s->sigmas[s->step_index - 1];
    double alpha_t = 1.0 - sigma_t;
    double h = lambda_of(sigma_t) - lambda_of(sigma_s0);
    double hh = -h;
    double h_phi_1 = expm1(hh);
    double b_h = expm1(hh); /* bh2 */
    const float *m0 = s->model_outputs[1];
    const float *x = s->last_sample;

    double c_x = sigma_t / sigma_s0;
    double c_m0 = -alpha_t * h_phi_1;

    if (order == 1) {
        double rho = 0.5;
        for (int64_t i = 0; i < n; i++) {
            double xt_ = c_x * x[i] + c_m0 * m0[i];
            this_sample[i] = (float)(xt_ - alpha_t * b_h * rho * (model_t[i] - m0[i]));
        }
    } else {
        double sigma_s1 = s->sigmas[s->step_index - 2];
        double r0 = (lambda_of(sigma_s1) - lambda_of(sigma_s0)) / h;
        const float *m1 = s->model_outputs[0];
        /* solve R rhos = b with R = [[1, 1], [r0, 1]] */
        double h_phi_k = h_phi_1 / hh - 1.0;
        double b1 = h_phi_k * 1.0 / b_h;
        h_phi_k = h_phi_k / hh - 1.0 / 2.0;
        double b2 = h_phi_k * 2.0 / b_h;
        double det = 1.0 * 1.0 - 1.0 * r0;
        double rho0 = (b1 - b2) / det;
        double rho1 = (b2 - r0 * b1) / det;
        for (int64_t i = 0; i < n; i++) {
            double xt_ = c_x * x[i] + c_m0 * m0[i];
            double corr = rho0 * (m1[i] - m0[i]) / r0 + rho1 * (model_t[i] - m0[i]);
            this_sample[i] = (float)(xt_ - alpha_t * b_h * corr);
        }
    }
}

static void unipc_step(bn_scheduler *s, const float *model_output, double t, float *sample, int64_t n) {
    if (s->step_index < 0) s->step_index = nearest_timestep_index(s, t);

    if (s->state_numel != n) {
        free_state(s);
        s->model_outputs[0] = (float *)calloc((size_t)n, sizeof(float));
        s->model_outputs[1] = (float *)calloc((size_t)n, sizeof(float));
        s->last_sample = (float *)calloc((size_t)n, sizeof(float));
        s->state_numel = n;
        s->lower_order_nums = 0;
    }

    int use_corrector = s->step_index > 0 && s->lower_order_nums > 0;

    /* convert_model_output for flow_prediction: x0 = sample - sigma * v */
    double sigma = s->sigmas[s->step_index];
    float *x0 = (float *)malloc(sizeof(float) * (size_t)n);
    for (int64_t i = 0; i < n; i++) x0[i] = (float)(sample[i] - sigma * model_output[i]);

    if (use_corrector)
        unipc_c_update(s, x0, sample, n, s->this_order);

    /* shift the model-output ring */
    memcpy(s->model_outputs[0], s->model_outputs[1], sizeof(float) * (size_t)n);
    memcpy(s->model_outputs[1], x0, sizeof(float) * (size_t)n);
    free(x0);

    int this_order = s->solver_order; /* lower_order_final */
    if (s->num_steps - s->step_index < this_order) this_order = s->num_steps - s->step_index;
    if (this_order > s->lower_order_nums + 1) this_order = s->lower_order_nums + 1;
    if (this_order < 1) this_order = 1;
    s->this_order = this_order;

    memcpy(s->last_sample, sample, sizeof(float) * (size_t)n);

    float *prev = (float *)malloc(sizeof(float) * (size_t)n);
    unipc_p_update(s, sample, prev, n, this_order);
    memcpy(sample, prev, sizeof(float) * (size_t)n);
    free(prev);

    if (s->lower_order_nums < s->solver_order) s->lower_order_nums++;
    s->step_index++;
}

void bn_sched_step(bn_scheduler *s, const float *model_output, double t,
                   float *sample, int64_t numel) {
    if (s->type == BN_SCHED_FLOW_MATCH)
        flow_match_step(s, model_output, t, sample, numel);
    else
        unipc_step(s, model_output, t, sample, numel);
}

double bn_sched_apg_sigma(const bn_scheduler *s, int t_idx) {
    if (s->type == BN_SCHED_UNIPC) {
        int idx = s->step_index < 0 ? 0 : s->step_index;
        return s->sigmas[idx];
    }
    return s->sigmas[t_idx];
}

void bn_sched_free(bn_scheduler *s) {
    free(s->sigmas);
    free(s->timesteps);
    free_state(s);
    s->sigmas = NULL;
    s->timesteps = NULL;
}
