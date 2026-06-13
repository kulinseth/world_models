# Bernini C port

A dependency-free C11 implementation of the Bernini Renderer inference
pipeline (`bernini/pipeline.py` → `BerniniRendererPipeline`). The model is
loaded once; each call generates one video / image. The complete inference
logic — tokenizer, UMT5 text encoder, dual-expert Wan2.2 DiT, flow-matching /
UniPC schedulers, APG & chained guidance, and the Wan causal-3D VAE — runs in
plain C on the CPU.

```
c/
├── include/bernini.h        public pipeline API  (pipeline.py: BerniniRendererPipeline)
├── src/
│   ├── pipeline.c           preprocess -> sample -> decode -> save  (pipeline.py)
│   ├── wan_diffusion.{h,c}  GEN_Wanx22 dual-expert sampler          (models/wan_diffusion.py)
│   ├── guidance.{h,c}       APG, MomentumBuffer, normalized guidance(models/wan_diffusion.py)
│   ├── transformer_wan.{h,c}WanTransformer3DModel + RoPE            (models/transformer_wan.py)
│   ├── attention.{h,c}      varlen attention, SDPA reference path   (attention.py)
│   ├── scheduler.{h,c}      FlowMatchScheduler + UniPC-bh2 (flow)   (models/scheduler.py, diffusers)
│   ├── wan_vae.{h,c}        AutoencoderKLWan encode/decode          (diffusers autoencoder_kl_wan.py)
│   ├── umt5.{h,c}           UMT5 encoder                            (transformers UMT5EncoderModel)
│   ├── tokenizer.{h,c}      SentencePiece unigram (spiece.model)    (AutoTokenizer)
│   ├── safetensors.{h,c}    sharded safetensors + EMA/prefix logic  (weights.py)
│   ├── data_utils.{h,c}     resize / normalize / frame selection    (data_utils.py)
│   ├── media.{h,c}          image & video IO via ffmpeg pipes       (io_utils.py, decord)
│   ├── bt.{h,c}             float32 tensor core (matmul, conv, norms)
│   ├── bjson.{h,c}          minimal JSON for configs & st headers
│   ├── rng.{h,c}            seeded MT19937 + gaussian noise
│   └── main.c               CLI                                     (cli.py + infer_single_gpu.py)
└── tests/test_core.c        unit tests for the numerics-critical pieces
```

## Build

```bash
cd c
make            # portable build -> ./bernini + libbernini.a
make BLAS=1     # macOS: link Accelerate for fast sgemm (strongly recommended)
make OMP=1      # OpenMP parallel loops (gcc; clang needs libomp)
make test       # run unit tests
```

`ffmpeg`/`ffprobe` on the PATH are used for image/video decode and H.264
encode (mirroring decord + imageio); `.ppm` in/out works without them.

## Run

The weight layout matches the Python side. Download a Bernini-R Diffusers
checkpoint (it bundles tokenizer/, text_encoder/, vae/, transformer/,
transformer_2/) and point `--config` at a config dir whose `config.json`
holds `wan22_base`, or directly at the self-contained Diffusers directory:

```bash
# text-to-video from the diffusers-format dir (transformer/transformer_2 loaded directly)
./bernini --config /path/to/Bernini-R-Diffusers \
  --prompt "a corgi surfing a wave at sunset" \
  --guidance_mode t2v --num_frames 33 --height 480 --width 832 \
  --output outputs/corgi.mp4

# video editing with separate Bernini Renderer checkpoints (EMA-aware loading)
./bernini --config configs/bernini_renderer_wan22 \
  --high_noise_ckpt /path/to/high_noise_ckpt --low_noise_ckpt /path/to/low_noise_ckpt \
  --prompt "replace the car with a horse-drawn carriage" \
  --guidance_mode rv2v --video input.mp4 --output outputs/edit.mp4
```

The CLI flags mirror `bernini/cli.py` (same names and defaults, including the
standard Wan2.2 negative prompt). All seven renderer guidance modes are
implemented: `rv2v`, `v2v`, `v2v_chain`, `t2v`, `r2v_apg`, `v2v_apg`,
`t2v_apg`.

## Library usage

```c
#include "bernini.h"

bernini_renderer_pipeline_t *pipe = bernini_renderer_from_pretrained(
    "/path/to/Bernini-R-Diffusers", NULL, NULL, /*use_unipc*/ -1,
    /*shift*/ 0, /*use_src_id_rotary_emb*/ -1);

bernini_gen_params_t g;
bernini_gen_params_init(&g);           /* pipeline.py defaults */
g.prompt = "a cat in the rain";
g.guidance_mode = "t2v";
g.output_path = "out.mp4";
bernini_renderer_generate(pipe, &g);   /* tokenizer -> UMT5 -> sample -> VAE -> mp4 */
bernini_renderer_free(pipe);
```

## What matches the Python pipeline

- Weight loading: sharded `.safetensors` (+ `index.json`), bf16/f16 → f32,
  the `diff_dec.transformer.` / `transformer.` / bare prefix candidates, and
  the prefer-`ema.` rule from `bernini/weights.py`.
- The full sampling loop of `GEN_Wanx22.sample`: combo assembly (∅ / V / I /
  VI with per-source-id rotary embeddings), the high→low-noise expert switch
  at `switch_dit_boundary` with `omega_scale` rescaling, all CFG/APG modes,
  per-frame norm thresholds, momentum buffers, and the packed
  `(t h w)(ph pw c)` latent layout.
- WanTransformer3DModel: fp64 rotary tables (t/h/w bands + source-id),
  rms_norm_across_heads QK norm, FP32LayerNorm semantics, per-block
  scale-shift modulation, gelu-tanh FFN.
- Schedulers: `FlowMatchScheduler` exactly; UniPC multistep (bh2, order 2,
  predict-x0, flow sigmas, terminal zero sigma, lower-order-final, corrector)
  as diffusers configures it for Wan2.2.
- Wan VAE: causal convs with the chunked feature-cache semantics (encode in
  1+4k frame chunks, decode frame by frame, first frame treated as an image),
  RMS norms, single-head spatial mid-block attention, latent mean/std
  normalization, `DiagonalGaussianDistribution.mode()`.
- Preprocessing: `MaxLongEdgeMinShortEdgeResize` (bicubic + antialias,
  stride snapping), `[-1,1]` normalization, `smart_video_nframes` frame
  selection, `make_divisible`.
- Output: H.264 mp4 via libx264 + yuv420p + crf 8 (single frames as png).

## Known differences

- Float32 everywhere (the Python pipeline mixes bf16/fp32); results are
  numerically close but not bit-identical.
- The noise RNG is MT19937 + Box–Muller, not torch's Philox/MT pipeline — a
  given `--seed` produces a different (equally valid) sample than Python.
- The T5 tokenizer skips the precompiled NFKC charsmap and `ftfy`/HTML
  unescaping; pass clean UTF-8 prompts.
- Bicubic antialias resize follows the torchvision kernel (a = -0.75); edge
  pixels can differ by ±1/255 from PIL.
- Single process, CPU only: the Ulysses sequence-parallel collectives in
  `bernini/parallel/` are no-ops on one rank and are not ported.
- The full Bernini family (Qwen2.5-VL semantic planner + MAR/FM vit decoder,
  `BerniniPipeline`) is not in the C port; this covers the renderer
  (`BerniniRendererPipeline`) end to end. The Wan2.2-TI2V residual VAE
  (`is_residual=True`) is likewise not supported.

## Performance notes

This is a reference port: correctness and structure first. `make BLAS=1
OMP=1` is the practical configuration; the 14B renderer at 480p×81f is still
hours-per-video on CPU — the 1.3B Bernini-R checkpoint at small sizes
(`--num_frames 9 --height 240 --width 320 --num_inference_steps 10`) is the
realistic smoke-test target.
