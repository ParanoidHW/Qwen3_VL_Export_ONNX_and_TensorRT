from dataclasses import dataclass


@dataclass
class ArgsConfig:
    """Configuration for Qwen3-VL model export ONNX"""
    # Model parameters
    qwen_path: str = '/home/wsj/Downloads/weights/qwen35-vl-2b'
    """Path to the qwen directory or directories"""

    export_path: str = 'export/qwen35_vl_2b'
    """Directory to save onnx model checkpoints."""

    batch_size: int = 1
    """Batch size of input for ONNX model inference"""

    imgs_paths: tuple = ("demo_data/input1.png", )
    """Path of images for ONNX model inference"""

    device: str = "auto"
    """Device used for ONNX export: auto, cuda, npu, or cpu."""

    dtype = 'fp16'
    """Data type of ONNX model: 'fp16' or 'fp32' """

    max_sequence_length: int = 512
    """Static token cache length for prefill/decode ONNX export."""

    decode_sequence_length: int = 1
    """Static decode step length. Keep this as 1 for token-by-token decoding."""

    export_parts: tuple = ("vit", "vlm", "llm_prefill", "llm_decode", "gen", "embed", "embed_select")
    """ONNX submodules to export for the Qwen3.5-VL inference chain."""

    verbose_export: bool = False
    """Print verbose torch.onnx export graph logs."""
