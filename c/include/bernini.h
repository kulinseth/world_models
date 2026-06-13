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

/* Public C pipeline API for the Bernini Renderer.
 *
 * Mirrors bernini/pipeline.py BerniniRendererPipeline:
 *
 *   BerniniRendererPipeline.from_pretrained(...) -> bernini_renderer_from_pretrained()
 *   pipeline(prompt, ...)                        -> bernini_renderer_generate()
 *
 * The model is loaded once; each generate call produces one video / image.
 * End-to-end flow: tokenize -> UMT5 encode -> VAE-encode visual conditions ->
 * dual-expert Wan2.2 guided sampling -> VAE decode -> save. */

#ifndef BERNINI_H
#define BERNINI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bernini_renderer_pipeline bernini_renderer_pipeline_t;

typedef struct {
    const char *prompt;            /* required */
    const char *neg_prompt;        /* "" if none */
    const char *system_prompt;     /* prefix added to the prompt ("") */
    int num_frames;                /* default 81 */
    int max_image_size;            /* default 624 (CLI default 848) */
    int height, width;             /* defaults 480 x 832; overridden by source media */
    const char *const *videos;     /* source video paths (editing), or NULL */
    int n_videos;
    const char *image;             /* single source image, or NULL */
    const char *const *images;     /* reference image paths, or NULL */
    int n_images;
    int num_inference_steps;       /* default 40 */
    const char *guidance_mode;     /* rv2v | v2v | v2v_chain | t2v | r2v_apg | v2v_apg | t2v_apg */
    double omega_vid, omega_img, omega_txt, omega_scale;
    double flow_shift;             /* default 5.0 */
    uint64_t seed;                 /* default 42 */
    int fps;                       /* default 16 */
    double eta;                    /* default 0.5 */
    double norm_threshold[2];      /* default {50, 50} */
    double momentum;               /* default -0.5 */
    const char *output_path;       /* default "output.mp4" */
    int write_output;              /* skip decode+save when 0 */
} bernini_gen_params_t;

/* Fill a params struct with the pipeline.py defaults. */
void bernini_gen_params_init(bernini_gen_params_t *p);

/* Load the renderer once. `config_dir` follows the Python layout: a config
 * directory whose config.json carries `wan22_base` (or a self-contained
 * Diffusers directory). `high_noise_ckpt` / `low_noise_ckpt` are Bernini
 * Renderer checkpoint dirs; pass NULL for both to load the transformers from
 * `wan22_base`'s transformer/ and transformer_2/ subfolders directly.
 * Returns NULL on failure. */
bernini_renderer_pipeline_t *bernini_renderer_from_pretrained(
    const char *config_dir,
    const char *high_noise_ckpt,
    const char *low_noise_ckpt,
    int use_unipc,            /* <0: keep config.json value */
    double shift_override,    /* <=0: keep config.json value */
    int use_src_id_rotary_emb /* <0: keep config.json value */);

/* Generate one clip / image; returns 0 on success. */
int bernini_renderer_generate(bernini_renderer_pipeline_t *pipe,
                              const bernini_gen_params_t *params);

void bernini_renderer_free(bernini_renderer_pipeline_t *pipe);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_H */
