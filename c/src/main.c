/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* Bernini Renderer single-process CLI (mirrors bernini/cli.py +
 * infer_single_gpu.py). */

#include "../include/bernini.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Standard Wan2.2 negative prompt (cli.py DEFAULT_NEG_PROMPT). */
static const char *DEFAULT_NEG_PROMPT =
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走";

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s --config DIR [options]\n"
        "\n"
        "model:\n"
        "  --config DIR              model config directory (config.json with wan22_base)\n"
        "  --high_noise_ckpt DIR     high-noise checkpoint dir (safetensors)\n"
        "  --low_noise_ckpt DIR      low-noise checkpoint dir (safetensors)\n"
        "  --use_unipc 0|1           scheduler choice (default: config.json)\n"
        "  --use_src_tgt_id 0|1      source-id rotary embeddings (default: config.json)\n"
        "\n"
        "input:\n"
        "  --prompt TEXT             text prompt / editing instruction (required)\n"
        "  --video PATH [PATH..]     source video path(s)\n"
        "  --image PATH              single source image\n"
        "  --images PATH [PATH..]    reference image(s)\n"
        "  --output PATH             output path (default outputs/output.mp4)\n"
        "\n"
        "generation:\n"
        "  --neg_prompt TEXT         (default: standard Wan2.2 negative prompt)\n"
        "  --system_prompt TEXT\n"
        "  --num_frames N            (81)\n"
        "  --max_image_size N        (848)\n"
        "  --height N --width N      (480 x 848)\n"
        "  --num_inference_steps N   (40)\n"
        "  --guidance_mode MODE      rv2v|v2v|v2v_chain|t2v|r2v_apg|v2v_apg|t2v_apg (rv2v)\n"
        "  --omega_vid F (1.25) --omega_img F (4.5) --omega_txt F (4.0)\n"
        "  --omega_scale F (0.8) --flow_shift F (5.0)\n"
        "  --seed N (42) --fps N (16) --eta F (0.5) --momentum F (0)\n"
        "  --norm_threshold F [F]    (50 50)\n",
        argv0);
}

static int is_flag(const char *a, const char *name) { return !strcmp(a, name); }

int main(int argc, char **argv) {
    const char *config = "configs/bernini_renderer_wan22";
    const char *high_ckpt = NULL, *low_ckpt = NULL;
    int use_unipc = -1, use_src_tgt_id = -1;
    const char *videos[16];
    const char *images[16];
    int n_videos = 0, n_images = 0;

    bernini_gen_params_t g;
    bernini_gen_params_init(&g);
    g.neg_prompt = DEFAULT_NEG_PROMPT;
    g.max_image_size = 848; /* cli.py default */
    g.width = 848;
    g.omega_vid = 1.25;
    g.omega_img = 4.5;
    g.omega_txt = 4.0;
    g.omega_scale = 0.8;
    g.momentum = 0.0;
    g.output_path = "outputs/output.mp4";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *next = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (is_flag(a, "--help") || is_flag(a, "-h")) { usage(argv[0]); return 0; }
        else if (is_flag(a, "--config") && next) config = argv[++i];
        else if (is_flag(a, "--high_noise_ckpt") && next) high_ckpt = argv[++i];
        else if (is_flag(a, "--low_noise_ckpt") && next) low_ckpt = argv[++i];
        else if (is_flag(a, "--use_unipc") && next) use_unipc = atoi(argv[++i]);
        else if (is_flag(a, "--use_src_tgt_id") && next) use_src_tgt_id = atoi(argv[++i]);
        else if (is_flag(a, "--prompt") && next) g.prompt = argv[++i];
        else if (is_flag(a, "--neg_prompt") && next) g.neg_prompt = argv[++i];
        else if (is_flag(a, "--system_prompt") && next) g.system_prompt = argv[++i];
        else if (is_flag(a, "--video")) {
            while (i + 1 < argc && argv[i + 1][0] != '-' && n_videos < 16)
                videos[n_videos++] = argv[++i];
        } else if (is_flag(a, "--image") && next) g.image = argv[++i];
        else if (is_flag(a, "--images")) {
            while (i + 1 < argc && argv[i + 1][0] != '-' && n_images < 16)
                images[n_images++] = argv[++i];
        } else if (is_flag(a, "--output") && next) g.output_path = argv[++i];
        else if (is_flag(a, "--num_frames") && next) g.num_frames = atoi(argv[++i]);
        else if (is_flag(a, "--max_image_size") && next) g.max_image_size = atoi(argv[++i]);
        else if (is_flag(a, "--height") && next) g.height = atoi(argv[++i]);
        else if (is_flag(a, "--width") && next) g.width = atoi(argv[++i]);
        else if (is_flag(a, "--num_inference_steps") && next) g.num_inference_steps = atoi(argv[++i]);
        else if (is_flag(a, "--guidance_mode") && next) g.guidance_mode = argv[++i];
        else if (is_flag(a, "--omega_vid") && next) g.omega_vid = atof(argv[++i]);
        else if (is_flag(a, "--omega_img") && next) g.omega_img = atof(argv[++i]);
        else if (is_flag(a, "--omega_txt") && next) g.omega_txt = atof(argv[++i]);
        else if (is_flag(a, "--omega_scale") && next) g.omega_scale = atof(argv[++i]);
        else if (is_flag(a, "--flow_shift") && next) g.flow_shift = atof(argv[++i]);
        else if (is_flag(a, "--seed") && next) g.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (is_flag(a, "--fps") && next) g.fps = atoi(argv[++i]);
        else if (is_flag(a, "--eta") && next) g.eta = atof(argv[++i]);
        else if (is_flag(a, "--momentum") && next) g.momentum = atof(argv[++i]);
        else if (is_flag(a, "--norm_threshold")) {
            int k = 0;
            while (i + 1 < argc && argv[i + 1][0] != '-' && k < 2)
                g.norm_threshold[k++] = atof(argv[++i]);
            if (k == 1) g.norm_threshold[1] = g.norm_threshold[0];
        } else {
            fprintf(stderr, "unknown argument: %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if (!g.prompt) {
        fprintf(stderr, "error: provide --prompt\n");
        usage(argv[0]);
        return 2;
    }
    if ((high_ckpt == NULL) != (low_ckpt == NULL)) {
        fprintf(stderr, "error: --high_noise_ckpt and --low_noise_ckpt must be given together\n");
        return 2;
    }
    g.videos = videos;
    g.n_videos = n_videos;
    g.images = images;
    g.n_images = n_images;

    bernini_renderer_pipeline_t *pipe = bernini_renderer_from_pretrained(
        config, high_ckpt, low_ckpt, use_unipc, g.flow_shift, use_src_tgt_id);
    if (!pipe) {
        fprintf(stderr, "error: failed to load pipeline from %s\n", config);
        return 1;
    }
    int rc = bernini_renderer_generate(pipe, &g);
    bernini_renderer_free(pipe);
    return rc == 0 ? 0 : 1;
}
