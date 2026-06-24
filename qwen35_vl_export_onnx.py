import argparse
import json
import os
import shutil


EXPORT_PARTS = ("vit", "vlm", "llm", "llm_prefill", "llm_decode", "gen", "embed")
DEFAULT_EXPORT_PARTS = ("vit", "vlm", "llm_prefill", "llm_decode", "gen", "embed")
REQUIRED_CHAIN_PARTS = {"vit", "vlm", "llm_prefill", "llm_decode", "gen", "embed"}


def parse_args():
    parser = argparse.ArgumentParser(description="Export static Qwen3.5-VL ONNX submodules.")
    parser.add_argument("--qwen-path", default=None, help="Path to the Qwen3.5-VL model directory.")
    parser.add_argument("--export-path", default=None, help="Directory to save exported ONNX submodules.")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default=None, help="ONNX export dtype.")
    parser.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default=None, help="Device used for tracing/export.")
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Static prefill/cache sequence length.")
    parser.add_argument("--decode-sequence-length", type=int, default=None, help="Static decode step length.")
    parser.add_argument(
        "--export-parts",
        nargs="+",
        default=None,
        choices=EXPORT_PARTS,
        help="Submodules to export. 'llm' is a compatibility alias for llm_prefill and llm_decode.",
    )
    parser.add_argument(
        "--image",
        dest="imgs_paths",
        action="append",
        default=None,
        help="Image path for tracing inputs. Pass multiple times for multiple images.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify exported ONNX artifacts after export.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose torch.onnx export graph logs.")
    return parser.parse_args()


def apply_cli_overrides(config, args):
    if args.qwen_path is not None:
        config.qwen_path = args.qwen_path
    if args.export_path is not None:
        config.export_path = args.export_path
    if args.dtype is not None:
        config.dtype = args.dtype
    if args.device is not None:
        config.device = args.device
    if args.max_sequence_length is not None:
        config.max_sequence_length = args.max_sequence_length
    if args.decode_sequence_length is not None:
        config.decode_sequence_length = args.decode_sequence_length
    if args.export_parts is not None:
        config.export_parts = _normalize_export_parts(args.export_parts)
    if args.imgs_paths is not None:
        config.imgs_paths = tuple(args.imgs_paths)
        config.batch_size = 1
    config.verify_export = args.verify
    config.verbose_export = args.verbose
    return config


def _normalize_export_parts(export_parts):
    if export_parts is None:
        export_parts = DEFAULT_EXPORT_PARTS
    if isinstance(export_parts, str):
        export_parts = (export_parts,)
    normalized = []
    for part in export_parts:
        if part == "llm":
            expanded = ("llm_prefill", "llm_decode")
        else:
            expanded = (part,)
        for expanded_part in expanded:
            if expanded_part not in normalized:
                normalized.append(expanded_part)
    return tuple(normalized)


def _validate_config(config):
    if config.dtype not in ("fp16", "fp32"):
        raise ValueError("dtype must be 'fp16' or 'fp32'.")
    if int(config.max_sequence_length) < 1:
        raise ValueError("max_sequence_length must be >= 1.")
    if int(config.decode_sequence_length) < 1:
        raise ValueError("decode_sequence_length must be >= 1.")
    if int(config.decode_sequence_length) > int(config.max_sequence_length):
        raise ValueError("decode_sequence_length must be <= max_sequence_length.")
    export_parts = _normalize_export_parts(getattr(config, "export_parts", DEFAULT_EXPORT_PARTS))
    unknown_parts = [part for part in export_parts if part not in EXPORT_PARTS]
    if unknown_parts:
        raise ValueError(f"Unsupported export_parts entries: {unknown_parts}. Allowed values: {EXPORT_PARTS}.")
    if ("llm_prefill" in export_parts) != ("llm_decode" in export_parts):
        raise ValueError("llm_prefill and llm_decode must be exported together for shared KV cache.")
    config.export_parts = export_parts


def _clone_inputs(inputs):
    cloned = []
    for item in inputs:
        if hasattr(item, "clone"):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def export_part_onnx(torch, qwen_model, opt_model, onnx_inputs, onnx_path, config, state_dict_getter=None):

    if config.dtype == torch.float16:
        opt_model.half()

    state_dict = state_dict_getter(qwen_model) if state_dict_getter is not None else qwen_model.state_dict()
    opt_model.load_state_dict(state_dict)
    opt_model = opt_model.to(config.device)
    opt_model.eval()

    torch.onnx.export(
        opt_model,
        _clone_inputs(onnx_inputs["inputs"]),
        onnx_path,
        input_names=onnx_inputs["input_names"],
        output_names=onnx_inputs["output_names"],
        dynamic_axes=onnx_inputs["dynamic_axes"],
        opset_version=18,
        do_constant_folding=False,
        external_data=True,
        verbose=getattr(config, "verbose_export", False),
    )
    return opt_model


def _resolve_runtime(torch, config):
    requested_device = getattr(config, "device", None) or "auto"
    if requested_device == "auto":
        config.device = _auto_export_device(torch)
    else:
        config.device = requested_device
    _validate_export_device(torch, config.device)

    requested_dtype = config.dtype
    config.dtype = torch.float16 if config.dtype == "fp16" else torch.float32
    if config.device == "cpu" and config.dtype == torch.float16:
        raise RuntimeError(
            "fp16 Qwen3.5 ONNX export requires CUDA or Ascend NPU. "
            "Use --dtype fp32 for CPU export, or pass --device cuda/npu on an accelerator machine."
        )
    print(f"Export device: {config.device}, dtype: {requested_dtype}")


def _auto_export_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if _torch_npu_is_available(torch):
        return "npu"
    return "cpu"


def _torch_npu_is_available(torch):
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "is_available"):
        return bool(npu.is_available())
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and hasattr(npu, "is_available") and npu.is_available())


def _validate_export_device(torch, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
    if device == "npu" and not _torch_npu_is_available(torch):
        raise RuntimeError(
            "--device npu was requested, but torch_npu/torch.npu is not available. "
            "Install the Ascend PyTorch adapter and run in an NPU environment."
        )
    if device != "cpu" and device not in ("cuda", "npu"):
        raise ValueError(f"Unsupported export device: {device!r}.")


def _shape_of(tensor):
    return [int(dim) for dim in tensor.shape]


def _as_tuple(outputs):
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _collect_output_shapes(torch, exported_models, onnx_input, export_parts):
    output_shapes = {}
    with torch.no_grad():
        for part in export_parts:
            model = exported_models[part]
            inputs = onnx_input[part]
            outputs = _as_tuple(model(*_clone_inputs(inputs["inputs"])))
            output_shapes[part] = {
                name: _shape_of(tensor)
                for name, tensor in zip(inputs["output_names"], outputs)
            }
    return output_shapes


def _collect_graph_shapes(onnx_path, export_parts):
    try:
        import onnx
    except ImportError:
        return {}, {}

    graph_input_shapes = {}
    graph_output_shapes = {}
    for part in export_parts:
        model_path = os.path.join(onnx_path, part, f"{part}.onnx")
        if not os.path.exists(model_path):
            continue
        model = onnx.load(model_path)
        graph_input_shapes[part] = {
            value_info.name: _value_info_shape(value_info)
            for value_info in model.graph.input
        }
        graph_output_shapes[part] = {
            value_info.name: _value_info_shape(value_info)
            for value_info in model.graph.output
        }
    return graph_input_shapes, graph_output_shapes


def _value_info_shape(value_info):
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _runtime_inputs_for_part(part, input_names):
    return list(input_names)


def _layout_inputs_for_part(part, input_names):
    if part == "vlm":
        return ["mm_token_type_ids", "image_grid_thw"]
    return []


def write_export_manifest(onnx_path, onnx_input, export_parts, config, input_shapes, output_shapes):
    manifest = {
        "model_type": "qwen3.5-vl",
        "static_shape": True,
        "external_data": True,
        "complete_chain": REQUIRED_CHAIN_PARTS.issubset(set(export_parts)),
        "max_sequence_length": int(config.max_sequence_length),
        "prompt_sequence_length": int(getattr(config, "prompt_sequence_length", config.max_sequence_length)),
        "decode_sequence_length": int(config.decode_sequence_length),
        "dtype": str(config.dtype).removeprefix("torch."),
        "vision_grid_thw": [list(row) for row in getattr(config, "vision_grid_thw", ())],
        "vision_spatial_merge_size": int(getattr(config, "vision_spatial_merge_size", 1)),
        "image_embed_lengths": [int(length) for length in getattr(config, "image_embed_lengths", ())],
        "parts": {},
        "cache": {
            "prefill": "llm_prefill",
            "decode": "llm_decode",
            "mapping": [],
        },
        "runtime_notes": {
            "vlm_attention_masks": "Static int mask with 1 for valid prompt tokens and 0 for padding.",
            "vlm_outputs": "Feed VLM position_ids, inputs_embeds, attention_mask, and linear_attention_mask to llm_prefill inputs with matching names.",
            "vlm_prompt_layout": "The VLM graph is traced for the exported static prompt layout: sequence length, image count, image grid, and image placeholder pattern are fixed by the export inputs. Layout inputs may still be graph inputs and must be provided with the same static shapes.",
            "decode_embedding": "Feed generated token ids to embed, then feed embed.inputs_embeds to llm_decode.inputs_embeds.",
            "external_data": "Large ONNX models are exported with external data files next to each part .onnx file.",
            "llm_prefill_attention_mask": "Static 4D additive mask. Fill causal and padding-key positions with dtype minimum.",
            "llm_prefill_linear_attention_mask": "Static 2D mask with 1 for valid prefill tokens and 0 for padding.",
            "llm_decode_attention_mask": "Static 4D additive mask over the full cache length for the decode token window.",
            "llm_decode_linear_attention_mask": "Static 2D mask, normally all ones for token-by-token decode.",
            "llm_decode_position_ids": "Static-shape decode position ids. Update values for the current decode token window at runtime while preserving the exported shape.",
            "cache_position": "Static int positions to write current tokens into the shared cache. Prefill traces all padded slots; decode cache_position should be set to the actual next token position at runtime.",
            "cache_handoff": "Feed each llm_prefill present_* output to the matching llm_decode past_* input; feed each decode present_* output back to the same decode past_* input for the next step.",
        },
    }

    for part in export_parts:
        inputs = onnx_input[part]
        manifest["parts"][part] = {
            "path": os.path.join(part, f"{part}.onnx"),
            "input_names": list(inputs["input_names"]),
            "output_names": list(inputs["output_names"]),
            "input_shapes": {
                name: _shape_of(tensor)
                for name, tensor in zip(inputs["input_names"], inputs["inputs"])
            },
            "graph_input_shapes": input_shapes.get(part, {}),
            "output_shapes": output_shapes.get(part, {}),
            "runtime_inputs": _runtime_inputs_for_part(part, inputs["input_names"]),
            "layout_inputs": _layout_inputs_for_part(part, inputs["input_names"]),
        }

    if "llm_prefill" in export_parts and "llm_decode" in export_parts:
        prefill_outputs = onnx_input.get("llm_prefill", {}).get("output_names", [])
        decode_input_shapes = manifest["parts"].get("llm_decode", {}).get("graph_input_shapes", {})
        decode_inputs = set(decode_input_shapes)
        if not decode_inputs:
            decode_inputs = set(onnx_input.get("llm_decode", {}).get("input_names", []))
        decode_output_shapes = output_shapes.get("llm_decode", {})
        decode_outputs = set(decode_output_shapes)
        if not decode_outputs:
            decode_outputs = set(onnx_input.get("llm_decode", {}).get("output_names", []))
        prefill_output_shapes = output_shapes.get("llm_prefill", {})
        if not decode_input_shapes:
            decode_input_shapes = manifest["parts"].get("llm_decode", {}).get("input_shapes", {})
        for present_name in prefill_outputs:
            if not present_name.startswith("present_"):
                continue
            past_name = "past_" + present_name.removeprefix("present_")
            manifest["cache"]["mapping"].append(
                {
                    "prefill_output": present_name,
                    "decode_input": past_name,
                    "decode_output": present_name,
                    "decode_input_exists": past_name in decode_inputs,
                    "decode_output_exists": present_name in decode_outputs,
                    "shape": prefill_output_shapes.get(present_name),
                    "decode_input_shape": decode_input_shapes.get(past_name),
                    "decode_output_shape": decode_output_shapes.get(present_name),
                }
            )

    os.makedirs(onnx_path, exist_ok=True)
    manifest_path = os.path.join(onnx_path, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Write ONNX manifest: {manifest_path}")



def run_export(config):
    _validate_config(config)

    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from modules.qwen35_vl import (
        Qwen35VLTextModelOpt,
        Qwen35VLTextModelWithCacheOpt,
        Qwen35VLVisualModelOpt,
        Qwen35VLModelOpt,
        Qwen35VLForConditionalGenerationOpt,
        Qwen35VLTokenEmbeddingOpt,
    )
    from utils import get_model_input, get_qwen35_onnx_input

    _resolve_runtime(torch, config)
    torch.manual_seed(42)

    model_input = get_model_input(config)
    qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(config.qwen_path, dtype=torch.float32, device_map='cpu', attn_implementation="eager")
    # res = qwen_model.generate(**model_input, use_cache=False)
    config.qwen_text_config = qwen_model.model.language_model.config
    config.qwen_vision_config = qwen_model.model.config.vision_config

    print("Init model load done!")
    onnx_input = get_qwen35_onnx_input(
        config,
        model_input,
        llm_hidden_size=qwen_model.model.language_model.config.hidden_size,
        vit_hidden_size=qwen_model.model.config.vision_config.out_hidden_size,
        gen_hidden_size=qwen_model.config.text_config.hidden_size,
    )

    print("Export ONNX model type: ", config.dtype)
    onnx_path = config.export_path + "/ONNX"

    part_qwen_model = {
        "llm": {
            "original": qwen_model.model.language_model,
            "optimized": Qwen35VLTextModelOpt(qwen_model.model.language_model.config),
        },
        "llm_prefill": {
            "original": qwen_model.model.language_model,
            "optimized": Qwen35VLTextModelWithCacheOpt(qwen_model.model.language_model.config, has_previous_state=False),
        },
        "llm_decode": {
            "original": qwen_model.model.language_model,
            "optimized": Qwen35VLTextModelWithCacheOpt(qwen_model.model.language_model.config, has_previous_state=True),
        },
        "vit": {
            "original": qwen_model.model.visual,
            "optimized": Qwen35VLVisualModelOpt(qwen_model.model.visual.config),
        },
        "vlm": {
            "original": qwen_model.model,
            "optimized": Qwen35VLModelOpt(qwen_model.model.config, config),
        },
        "gen": {
            "original": qwen_model,
            "optimized": Qwen35VLForConditionalGenerationOpt(qwen_model.config),
            "state_dict_getter": lambda model: {
                f"lm_head.{key}": value for key, value in model.lm_head.state_dict().items()
            },
        },
        "embed": {
            "original": qwen_model.model.language_model,
            "optimized": Qwen35VLTokenEmbeddingOpt(qwen_model.model.language_model.config),
            "state_dict_getter": lambda model: {
                f"embed_tokens.{key}": value for key, value in model.embed_tokens.state_dict().items()
            },
        },
    }


    export_parts = _normalize_export_parts(getattr(config, "export_parts", tuple(part_qwen_model.keys())))
    exported_models = {}
    for part_model_name in export_parts:
        # remove the previous dir
        onnx_part_dir = os.path.join(onnx_path, part_model_name)
        if os.path.exists(onnx_part_dir):
            shutil.rmtree(onnx_part_dir)
        os.makedirs(onnx_part_dir)

        exported_models[part_model_name] = export_part_onnx(
            torch=torch,
            qwen_model=part_qwen_model[part_model_name]["original"],
            opt_model=part_qwen_model[part_model_name]["optimized"],
            onnx_inputs=onnx_input[part_model_name],
            onnx_path=os.path.join(onnx_path, part_model_name, part_model_name + ".onnx"),
            config=config,
            state_dict_getter=part_qwen_model[part_model_name].get("state_dict_getter"),
        )
        print(f"Export {part_model_name} ONNX model done!")

    graph_input_shapes, output_shapes = _collect_graph_shapes(onnx_path, export_parts)
    if not output_shapes:
        output_shapes = _collect_output_shapes(torch, exported_models, onnx_input, export_parts)
    write_export_manifest(onnx_path, onnx_input, export_parts, config, graph_input_shapes, output_shapes)
    if getattr(config, "verify_export", False):
        from verify_qwen35_onnx_exports import verify_exports

        verify_exports(onnx_path, int(config.max_sequence_length))
    print("Export Qwen35 done!")


if __name__ == "__main__":
    args = parse_args()
    from config.qwen35_config import ArgsConfig

    cfg = ArgsConfig()
    cfg = apply_cli_overrides(cfg, args)
    run_export(cfg)
