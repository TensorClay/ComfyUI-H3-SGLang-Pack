# ComfyUI H3 SGLang Pack

ComfyUI H3 SGLang Pack is a collection of drop-in replacement nodes for accelerating [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) video generation by distributing compute through SGLang. It supports ComfyUI’s current self-hosted H3 text-to-video, first/last-frame image-to-video, and reference-to-video workflows while retaining the standard conditioning, sampling, scheduling, and decode nodes.

The loader keeps ComfyUI’s normal generation controls while moving H3 denoiser evaluations to a local SGLang TP/Ulysses worker pool. In paired benchmarks, generation was 3.17× faster for a light workload and 8.21× faster for a heavy workload compared with warm native runs. Results depend on hardware, topology, attention backend, and workload.

If you couldn't tell from the tone of the README, this whole thing was vibe-coded. Use at your own risk!

## What is SGLang?

[SGLang](https://www.sglang.io/) is an open-source serving and execution framework for language and multimodal models. Its diffusion runtime provides the distributed tensor and sequence parallelism used by this pack.

For SGLang’s broader image and video integration, including Server and Integrated modes, see the official [ComfyUI_SGLDiffusion project](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion).

## Benchmarks

The native and SGLang tests used identical generation settings and differed only in the diffusion-model loader. Every output was reviewed in ComfyUI and confirmed to look and sound correct.

| Workload | Model | Target resolution | Duration | Steps | Image references | Audio references |
|---|---|---:|---:|---:|---:|---:|
| Light | Ref2VA BF16 | 864×480, 0.4 MP, 16:9 | 5 seconds | 20 | 2 | 0 |
| Heavy | Ref2VA BF16 | 1920×1080, 2.0 MP, 16:9 | 20 seconds | 20 | 4 | 1 |

| Workload | Run | Native ComfyUI | SGLang TP2/Ulysses4 | Speedup | Time reduction |
|---|---|---:|---:|---:|---:|
| Light | Cold | 3:05 | 1:51.78 | 1.66× | 39.6% |
| Light | Warm | 1:58 | 0:37.33 | 3.17× | 68.4% |
| Heavy | Cold | 4:18:20 | 32:46 | 7.88× | 87.3% |
| Heavy | Warm | 4:17:09 | 31:19 | 8.21× | 87.8% |

The native tests used one GPU. The SGLang tests used eight GPUs at TP2/Ulysses4. The native measurements are the original baseline. The automatic-attention light measurements were repeated after adding the attention, LoRA, caching, and all-model-flow integrations; the heavy SGLang measurements are from the preceding release audit. Cold SGLang runs include worker startup, while warm runs reuse the runtime.

The same light workflow was also measured with each worker attention selection. These are end-to-end times, including conditioning, denoising, both VAE decodes, and video encoding:

| Attention selection | Cold | Warm | Cold speedup vs native | Warm speedup vs native |
|---|---:|---:|---:|---:|
| Automatic | 1:51.78 | 0:37.33 | 1.66× | 3.17× |
| SGLang Flash Attention | 2:06.96 | 0:37.49 | 1.46× | 3.15× |
| Sage Attention | 2:10.34 | 0:37.34 | 1.42× | 3.16× |
| Sol-Attn | 1:50.73 | 0:35.82 | 1.67× | 3.30× |



## Prerequisites

SGLang Diffusion installed in the **same Python environment as ComfyUI**. Follow the official [SGLang Diffusion installation documentation](https://docs.sglang.io/docs/sglang-diffusion) for your system.

Optional features have their own dependencies:

- **Cache-DiT:** install [`cache-dit`](https://github.com/vipshop/cache-dit).
- **Sage Attention:** install [`sageattention`](https://github.com/thu-ml/SageAttention). The `sageattn3` choices additionally require the separately built [SageAttention3 package](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/README.md) and a supported Blackwell GPU. The KJNodes **MiniMax H3 Mem Eff Sage Attention Patch** chain requires SageAttention 2.2.0 or newer and [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).
- **Sol Attention:** install [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) beside this pack in `ComfyUI/custom_nodes`.

## Installation

### ComfyUI Registry / Manager

Open ComfyUI’s extension manager, search for **H3 SGLang Pack**, install it, and restart ComfyUI. SGLang itself must already satisfy the prerequisite above.

### Git

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/TensorClay/ComfyUI-H3-SGLang-Pack.git
```

Restart ComfyUI after cloning.

## Nodes

### Load MiniMax H3 Diffusion Model (SGLang)

A drop-in replacement for **Load Diffusion Model** in Comfy-Org’s official [text-to-video](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json), [image-to-video](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json), and [reference-to-video](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) workflows. It returns the standard ComfyUI `MODEL` type, so the existing H3 conditioning, guider, sampler, scheduler, latent, and VAE decode nodes remain in control of their usual stages.

| Parameter | Description |
|---|---|
| `model_name` | A compatible H3 `.safetensors` checkpoint in `ComfyUI/models/diffusion_models`. Full BF16, pruned BF16, and ComfyUI pruned INT8 ConvRot FL2VA, Ref2VA, and hybrid exports are supported. The filename must contain `fl2va`, `ref2va`, or `hybrid`. |
| `topology` | A compatible TP/Ulysses layout using some or all accelerators visible to ComfyUI. Options are generated from the current device count and H3’s sharding constraints. |
| `hybrid_mode` | Appears only for a checkpoint whose filename contains `hybrid`. Select `ref2va` or `fl2va` to tell SGLang which model path to initialize and warm up. |

Choose the FL2VA checkpoint for text-to-video and first/last-frame image-to-video. Choose the Ref2VA checkpoint for image, video, and audio references. Connect the loader’s `MODEL` output where the stock loader was connected; the rest of the official graph can remain unchanged.

The first generation starts the local SGLang workers. Consecutive compatible SGLang generations reuse the runtime and loaded checkpoint. Before a queued workflow that does not use this pack’s loader begins, the pack unloads its workers so native ComfyUI models can reclaim their VRAM. Changing the checkpoint or topology also replaces the runtime.

Under the hood, ComfyUI retains the sampling loop and sends the current H3 latent streams to SGLang for each denoiser evaluation. Prompt encoding, scheduling, VAE decoding, and output encoding remain in ComfyUI. Whole-model ComfyUI diffusion wrappers still execute in the host process around the distributed call. Worker-local ControlNet tensors and arbitrary transformer patch callbacks are not transportable through the standard `MODEL` interface; use the worker-native attention, LoRA, and Cache-DiT adapters supplied by this pack.

### Attention patch nodes

ComfyUI attention patch callbacks run in the host process and cannot be transported into SGLang’s worker processes. This pack therefore provides worker-native replacements with the same input and output contracts as the common ecosystem nodes:

- **Patch Sage Attention KJ (SGLang)** — replacement for [KJNodes’ Patch Sage Attention KJ](https://github.com/kijai/ComfyUI-KJNodes)
- **Patch Sol-Attn (SGLang)** — replacement for [ComfyUI-SolAttn_triton’s Patch Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton)
- **Patch Flash Attention DN (SGLang)** — replacement for [ComfyUI-DN_PatchFlashAttention’s Patch Flash Attention DN](https://github.com/0xDELUXA/ComfyUI-DN_PatchFlashAttention)

Replace the corresponding stock patch node while keeping its existing links and widget values.

#### Patch Sage Attention KJ (SGLang)

Selects a worker-native Sage Attention implementation. `sage_attention` mirrors KJNodes’ implementation choices, while optional `allow_compile` permits SageAttention’s compile path. `disabled` passes the input model through unchanged.

**Sage compatibility note:** Existing workflows may keep KJNodes’ **MiniMax H3 Mem Eff Sage Attention Patch** after **Patch Sage Attention KJ (SGLang)**, but the distributed path does not benefit from it. That node patches native H3 blocks in the ComfyUI process, while the actual blocks run inside SGLang workers using this pack’s worker-native Sage implementation. For new SGLang workflows, connect this pack’s Sage node directly to the sampler.

#### Patch Sol-Attn (SGLang)

Runs Sol-Attn inside each worker. Its `tau`, activation window, minimum token count, INT8 QK/PV controls, conditioning-sink mode, TMA option, dense-block list, verbosity, and optional per-block `tau_profile` mirror the upstream node. Morton token reordering is not yet supported and fails clearly when enabled.

#### Patch Flash Attention DN (SGLang)

The `enabled` switch mirrors the upstream node. When enabled, the replacement selects SGLang’s hardware-supported Flash backend inside the workers rather than transporting the upstream package’s Flash Attention 2 callback. On the validated SGLang 0.5.17/A100 environment this resolves to Flash Attention 3.

Omitting an attention patch lets SGLang choose its dense backend automatically. Switching attention settings restarts the worker pool.

#### Upstream compatibility and credit

These worker-native replacements mirror the public node contracts maintained by other ComfyUI contributors. Their current compatibility targets are:

- **Patch Sage Attention KJ** and **MiniMax H3 Mem Eff Sage Attention Patch** from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), maintained by [Kijai](https://github.com/kijai), at [`3f200542`](https://github.com/kijai/ComfyUI-KJNodes/commit/3f20054214fec9f9234fd3841ae6f1e4287948f6).
- **Patch Sol-Attn** from [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton), maintained by [Kijai](https://github.com/kijai), at [`842c4eaa`](https://github.com/kijai/ComfyUI-SolAttn_triton/commit/842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc).
- **Patch Flash Attention DN** from [ComfyUI-DN_PatchFlashAttention](https://github.com/0xDELUXA/ComfyUI-DN_PatchFlashAttention), maintained by [0xDELUXA](https://github.com/0xDELUXA), at [`b005549a`](https://github.com/0xDELUXA/ComfyUI-DN_PatchFlashAttention/commit/b005549ad33193bc0e946c9389de792dc767f5bc).

TensorClay is not the author or maintainer of those upstream packages, and this project is not affiliated with them. Later upstream releases may change their APIs or behavior before this pack is updated. Thank you to Kijai and 0xDELUXA for their contributions to the ComfyUI community.

### Load LoRA (SGLang)

A drop-in replacement for **Load LoRA** that applies a transformer LoRA on the SGLang workers. It keeps the stock `MODEL` input/output and parameter contract. Connect one or more nodes between the SGLang model loader and the guider to stack adapters. This direct path avoids rewriting a compatible adapter.

| Parameter | Description |
|---|---|
| `model` | A model from **Load MiniMax H3 Diffusion Model (SGLang)**, or another node in this pack that returns one. |
| `lora_name` | A LoRA discovered in `ComfyUI/models/loras`. |
| `strength_model` | Transformer adapter strength. Negative and greater-than-one values are allowed. |

Adapter tensors remain cached with the active worker runtime. Changing or removing the LoRA restores the base transformer weights before the next generation. ComfyUI’s stock **Load LoRA** and **Load LoRA Model Only** nodes also work with linear H3 LoRA, LoHa, and LoKr adapters; the pack translates those patches exactly into a temporary worker LoRA. DoRA, convolutional or reshaped adapters, and dense diff/set weight patches are not supported.

### MiniMax H3 Cache-DiT (SGLang)

Enables SGLang’s Cache-DiT integration for H3. Place it between the model loader or LoRA node and the guider. Cache-DiT skips eligible block computation when residual changes are small, trading exact output parity for speed.

| Parameter | Description |
|---|---|
| `model` | An H3 SGLang model. |
| `max_warmup_steps` | Full-compute steps before cache reuse can begin. |
| `residual_diff_threshold` | Maximum residual change allowed for reuse; higher values cache more aggressively. |
| `max_continuous_cached_steps` | Maximum consecutive reused steps before a full refresh. |

Cache state is scoped to one sampling execution and is released when that execution ends. Cache setup has a fixed cost, so benchmark it on the workload you intend to run.

Compatibility was validated with `cache-dit` 1.3.0 on the heavy Ref2VA workload described above. Both tested presets completed with correct video and audio, but neither improved performance:

| Cache-DiT setting | Warm end-to-end | Denoising | Change vs no cache |
|---|---:|---:|---:|
| Disabled | 31:16.56 | 28:18 | Baseline |
| W=4, R=0.04, MC=1 | 31:18.74 | 28:19 | 0.1% slower |
| W=4, R=0.24, MC=3 | 31:28.11 | 28:24 | 0.6% slower |

These differences are within normal run-to-run variation and show no measurable Cache-DiT benefit for this tested H3 workload. Other workloads, settings, Cache-DiT releases, and hardware may behave differently.
