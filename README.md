# Qwen3/3.5-ONNX: Image-to-Text Inference Model

## Overview
This repository provides the ONNX-converted version of the **qwen3-vl-2b** multimodal model, optimized for efficient image-to-text generation. The model supports inference on single images with a fixed input resolution of 224×224 and outputs descriptive text based on visual content.

## Key Features
- **Model Type**: ONNX-exported multimodal large language model (vision-language)
- **Input Specification**: Single RGB image (224×224 resolution, 3 channels)
- **Output**: High-quality natural language description of the input image
- **Conversion Source**: Original qwen3-vl-2b (PyTorch) → ONNX format

## Future Features
- **Qwen3.5-VL ONNX**: Qwen3.5 can be converted to ONNX, but its inference speed is slower than PyTorch. This is mainly because the torch_chunk_gated_delta_rule function in Qwen3.5 uses a large number of dynamic slicing operations and loops, resulting in a very large static computation graph in ONNX.

## Inference Example
### Input
<img src="demo_data/input1.png" alt="Input image">

- **Version**: Single RGB image (224×224) of a lemon.
- **Language**: Describe this image.

### Output
```
This image shows a single, yellow, spherical object that appears to be a small, smooth, and rounded lemon. It is placed on a light-colored, possibly white or off-white, surface with a wood grain texture. The lemon has a rounded, slightly flattened top and a smooth surface. The lighting is even, and the object is the central focus of the image.
```
## Next task
- Adapt images of different sizes
- Comparison of Test Torch and ONNX inference Speed
- Convert ONNX to TensorRT to further improve inference speed
- Convert more models from Torch to ONNX


## Usage
### 0. Environment Setup
```bash
conda create -n onnx python=3.10 -y
conda activate onnx
pip install -r requirements.txt
git clone https://github.com/garlic-byte/Qwen3_VL_Export_ONNX_and_TensorRT.git
cd Qwen3_VL_Export_ONNX_and_TensorRT
```

### 1. Download Qwen3-VL
```bash
# Download the model
mkdir qwen3-vl-2b
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir=qwen3-vl-2b/
```

### 2. Conert Torch to ONNX and Test Inference
```bash
python qwen3_vl_export_onnx.py
python inference_onnx.py
```

### 2.1 Export Qwen3.5-VL Static ONNX Submodules
Export with static shapes. `--max-sequence-length` controls the padded prefill/cache length, and `--decode-sequence-length` controls the decode token count:
```bash
python qwen35_vl_export_onnx.py --max-sequence-length 512 --decode-sequence-length 1 --verify
```
Defaults are also available in `config/qwen35_config.py` if you want to persist local settings, but the CLI arguments take precedence for each export run.
The default dtype is fp16 and requires CUDA; use `--dtype fp32` for CPU export.
By default the exporter writes the complete `vit -> vlm -> llm_prefill/llm_decode -> gen` chain plus `embed` for decode token embedding. Use `--export-parts` only when you intentionally want a partial export.
The verifier can also be run separately after export:
```bash
python verify_qwen35_onnx_exports.py
```
The Qwen3.5 export writes static ONNX submodules under `ONNX/` plus `ONNX/manifest.json`.
Large submodules are exported with ONNX external data files next to each `.onnx`; keep each part directory together when moving artifacts.
The manifest records static shapes, runtime inputs, layout inputs, and prefill/decode cache handoff names.
For `vlm`, the exported prompt layout is static: sequence length, image count/grid, and image placeholder pattern are fixed by the export inputs.
Layout inputs such as `mm_token_type_ids` and `image_grid_thw` can still be ONNX graph inputs, so provide them with the same static shapes recorded in the manifest.
The `vlm` graph outputs `position_ids`, `inputs_embeds`, `attention_mask`, and `linear_attention_mask`, which can be fed to `llm_prefill` inputs with the same names.
For decode, feed generated token ids to `embed`, then feed `embed.inputs_embeds` to `llm_decode.inputs_embeds`.
For each decode call, update `llm_decode.position_ids` and `cache_position` values for the current token window while keeping their exported static shapes.
The verifier checks static graph dimensions and that every `present_*` cache output can be fed to the matching `past_*` decode input.

### 3. Conert ONNX to TensorRT and Test Inference
## My environment
- CUDA 12.8
- TensorRT Debian local repo:
nv-tensorrt-local-repo-ubuntu2404-10.9.0-cuda-12.8_1.0-1_amd64.deb
- Python TensorRT wheel:
tensorrt-10.9.0.34

```bash
bash build_engine.sh
python inference_trt.py
```

## Model Conversion Notes
- The ONNX model is exported from the original PyTorch implementation of qwen3-vl-2b.
- Input resolution is fixed at 224×224 (consistent with the model's training configuration).
- For optimal performance, use ONNX Runtime with GPU acceleration (install `onnxruntime-gpu` instead of `onnxruntime`).
- The model retains the original qwen3-vl-2b's visual understanding and text generation capabilities.


## Performance Benchmark
| **Metric** | **Type** | **Value** |
|------------|----------|-----------|
| **Latency (1000 runs)** | Torch (fp32) | 44.46 (sec) |
| | ONNX (fp32) | 26.78 (sec) |
| | ONNX (fp16) | 18.13 (sec) |
| | TensorRT (fp16) | 13.77 (sec) |
| **Generation Speed (10 runs, fp16)** | Qwen3-vl | 19.378385 (tokens/sec) <br>*(Tokens generated: 1103)* |
| | ONNX (tokens/sec) | 38.667467 (tokens/sec) <br>*(Tokens generated: 1062)* |
| | TensorRT (tokens/sec) | 66.579019 (tokens/sec) <br>*(Tokens generated: 842)* |



## License
The model is licensed under the same license as the original qwen3-vl-2b (see [Qwen Official Repository](https://github.com/QwenLM/Qwen) for details).

## Acknowledgements
- Original qwen3-vl-2b model developed by Alibaba Cloud.
- ONNX conversion leverages PyTorch's `torch.onnx.export` API and ONNX Runtime for inference optimization.
