/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Schedulers for Wan sampling (inference only).
 * - BN_SCHED_FLOW_MATCH mirrors bernini/models/scheduler.py (FlowMatchScheduler)
 * - BN_SCHED_UNIPC mirrors diffusers UniPCMultistepScheduler configured the way
 *   Wan2.2 ships it: prediction_type="flow_prediction", use_flow_sigmas=True,
 *   predict_x0=True, solver_type="bh2", solver_order=2, lower_order_final=True,
 *   final_sigmas_type="zero". */

#ifndef BERNINI_SCHEDULER_H
#define BERNINI_SCHEDULER_H

#include "bt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { BN_SCHED_FLOW_MATCH, BN_SCHED_UNIPC } bn_sched_type;

typedef struct {
    bn_sched_type type;
    int    num_train_timesteps;
    double shift;
    double sigma_max, sigma_min;
    int    extra_one_step;

    int     num_steps;
    double *sigmas;    /* FlowMatch: [num_steps]; UniPC: [num_steps+1] (final 0) */
    double *timesteps; /* [num_steps] */

    /* UniPC multistep state */
    int    solver_order; /* fixed to 2 */
    int    step_index;   /* -1 == uninitialized */
    int    lower_order_nums;
    int    this_order;
    float *model_outputs[2]; /* converted x0 predictions, ring buffer */
    float *last_sample;
    int64_t state_numel;
} bn_scheduler;

void bn_sched_init(bn_scheduler *s, bn_sched_type type, int num_train_timesteps,
                   double shift, double sigma_min, int extra_one_step);

/* (Re)build the sigma/timestep tables; shift_override > 0 replaces shift.
 * Also resets all multistep state. */
void bn_sched_set_timesteps(bn_scheduler *s, int num_inference_steps, double shift_override);

/* One scheduler step: updates `sample` in place from the model output at
 * timestep `t` (both length `numel`). */
void bn_sched_step(bn_scheduler *s, const float *model_output, double t,
                   float *sample, int64_t numel);

/* Noise level used to convert v-pred to x-pred for APG guidance
 * (mirrors GEN_Wanx22._apg_sigma). */
double bn_sched_apg_sigma(const bn_scheduler *s, int t_idx);

void bn_sched_free(bn_scheduler *s);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_SCHEDULER_H */
