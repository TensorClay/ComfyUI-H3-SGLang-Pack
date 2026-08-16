# ComfyUI H3 SGLang Pack

ComfyUI H3 SGLang Pack is a collection of ComfyUI custom nodes for accelerating [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) video generation by distributing selected operations through SGLang. The pack currently contains one MiniMax H3 Ref2VA diffusion-model loader.

The loader keeps ComfyUI's normal generation controls while moving H3 denoiser evaluations to a local SGLang TP/Ulysses worker pool. In paired benchmarks, generation was 4.02× faster for a light workload and 8.21× faster for a heavy workload compared with warm native runs. Results depend on hardware, topology, and workload.

## What is SGLang?

[SGLang](https://www.sglang.io/) is an open-source serving and execution framework for language and multimodal models. Its diffusion runtime provides the distributed tensor and sequence parallelism used by this pack.

For SGLang's broader image and video integration, including Server and Integrated modes, see the official [ComfyUI_SGLDiffusion project](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion).

## Benchmarks

The native and SGLang tests used identical generation settings and differed only in the diffusion-model loader. Every output was reviewed in ComfyUI and confirmed to look and sound correct.

| Workload | Model | Target resolution | Duration | Steps | Image references | Audio references |
|---|---|---:|---:|---:|---:|---:|
| Light | Ref2VA BF16 | 864×480, 0.4 MP, 16:9 | 5 seconds | 20 | 2 | 0 |
| Heavy | Ref2VA BF16 | 1920×1080, 2.0 MP, 16:9 | 20 seconds | 20 | 4 | 1 |

| Workload | Run | Native ComfyUI | SGLang TP2/Ulysses4 | Speedup | Time reduction |
|---|---|---:|---:|---:|---:|
| Light | First | 3:05 | 0:29 | 6.30× | 84.1% |
| Light | Warm | 1:58 | 0:29 | 4.02× | 75.1% |
| Heavy | First | 4:18:20 | 32:46 | 7.88× | 87.3% |
| Heavy | Warm | 4:17:09 | 31:19 | 8.21× | 87.8% |

The native tests used one GPU. The SGLang tests used eight GPUs at TP2/Ulysses4. The native measurements are the original baseline; the SGLang measurements were repeated after the release audit and performance fixes.

The selected light SGLang run took 29.42 seconds; two adjacent repeats took 29.45 and 29.37 seconds. Because the runtime was already resident and no meaningful first/warm difference appeared, the selected result is shown in both light comparison rows. The first heavy SGLang run includes worker startup, while its warm run reused the runtime. An interrupted native heavy attempt produced no output and is excluded.

## Prerequisites

SGLang Diffusion installed in the **same Python environment as ComfyUI**. Follow the official [SGLang Diffusion installation documentation](https://docs.sglang.io/docs/sglang-diffusion) for your system.

## Installation

### ComfyUI Registry / Manager

Open ComfyUI's extension manager, search for **H3 SGLang Pack**, install it, and restart ComfyUI. SGLang itself must already satisfy the prerequisite above.

### Git

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/TensorClay/ComfyUI-H3-SGLang-Pack.git
```

Restart ComfyUI after cloning.

## Nodes

### Load MiniMax H3 Diffusion Model (SGLang)

A drop-in replacement for **Load Diffusion Model** in Comfy-Org's [official MiniMax H3 Ref2VA workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json). It returns the standard ComfyUI `MODEL` type, so the existing conditioning, guider, sampler, scheduler, latent, and VAE decode nodes remain in control of their usual stages.

#### Parameters

| Parameter | Description |
|---|---|
| `model_name` | A compatible H3 Ref2VA `.safetensors` checkpoint discovered in `ComfyUI/models/diffusion_models`. Full BF16 and ComfyUI's pruned INT8 ConvRot exports are supported. |
| `topology` | A compatible TP/Ulysses layout using some or all of the CUDA/ROCm GPUs visible to ComfyUI. The default uses the largest supported GPU count and prefers TP2 when possible. |

#### Usage

1. Place a supported checkpoint in `ComfyUI/models/diffusion_models`.
2. Open the official MiniMax H3 Ref2VA workflow.
3. Replace **Load Diffusion Model** with this node and connect its `MODEL` output to the existing graph.
4. Select the checkpoint and topology, then queue the workflow.

The first load starts the local SGLang workers. Later generations reuse that runtime until ComfyUI unloads the model to free VRAM or the model or topology selection changes.

Under the hood, ComfyUI retains the sampling loop and sends the current H3 latent streams to SGLang for each denoiser evaluation, receiving the predictions back afterward. Prompt encoding, scheduling, VAE decoding, and output encoding remain in ComfyUI. This boundary preserves granular workflow control, but the transfer cost means speedups vary with workload.

Current scope is MiniMax H3 Ref2VA. Arbitrary model patches, LoRAs, and ControlNet-style controls are not supported by the distributed adapter.
