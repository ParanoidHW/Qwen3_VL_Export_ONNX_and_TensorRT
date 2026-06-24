import argparse
import json
import os
from dataclasses import dataclass


REQUIRED_PARTS = ("vit", "vlm", "llm_prefill", "llm_decode", "gen", "embed")


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]


def _tensor_shape(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"{value_info.name} has no static tensor shape.")

    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            raise ValueError(f"{value_info.name} has dynamic dim_param={dim.dim_param!r}.")
        if dim.dim_value <= 0:
            raise ValueError(f"{value_info.name} has dynamic or unknown dim_value={dim.dim_value}.")
        dims.append(dim.dim_value)
    return tuple(dims)


def _load_infos(onnx_file):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("The onnx package is required to verify exported ONNX files.") from exc

    model = onnx.load(onnx_file)
    inputs = [TensorInfo(node.name, _tensor_shape(node)) for node in model.graph.input]
    outputs = [TensorInfo(node.name, _tensor_shape(node)) for node in model.graph.output]
    return inputs, outputs


def _expected_file(onnx_dir, part):
    return os.path.join(onnx_dir, part, f"{part}.onnx")


def _require_parts(onnx_dir, parts):
    missing = []
    for part in parts:
        onnx_file = _expected_file(onnx_dir, part)
        if not os.path.exists(onnx_file):
            missing.append(onnx_file)
    if missing:
        raise FileNotFoundError("Missing required ONNX files:\n" + "\n".join(missing))


def _info_by_name(infos):
    return {info.name: info for info in infos}


def _require_info(info_map, name, label):
    if name not in info_map:
        raise ValueError(f"{label} missing required tensor {name!r}.")


def _verify_static_parts(onnx_dir, parts):
    part_infos = {}
    for part in parts:
        onnx_file = _expected_file(onnx_dir, part)
        inputs, outputs = _load_infos(onnx_file)
        part_infos[part] = (inputs, outputs)
        print(f"{part}: {len(inputs)} inputs, {len(outputs)} outputs")
    return part_infos


def _verify_cache_pairing(part_infos):
    prefill_inputs, prefill_outputs = part_infos["llm_prefill"]
    decode_inputs, decode_outputs = part_infos["llm_decode"]

    prefill_output_map = _info_by_name(prefill_outputs)
    decode_input_map = _info_by_name(decode_inputs)
    decode_output_map = _info_by_name(decode_outputs)

    required_prefix_inputs = (
        "position_ids",
        "inputs_embeds",
        "attention_mask",
        "linear_attention_mask",
        "cache_position",
    )
    for name in required_prefix_inputs:
        if name not in decode_input_map:
            raise ValueError(f"llm_decode missing required input {name!r}.")
        if name not in _info_by_name(prefill_inputs):
            raise ValueError(f"llm_prefill missing required input {name!r}.")

    present_names = [info.name for info in prefill_outputs if info.name.startswith("present_")]
    if not present_names:
        raise ValueError("llm_prefill has no present_* cache outputs.")

    for present_name in present_names:
        past_name = "past_" + present_name.removeprefix("present_")
        if past_name not in decode_input_map:
            raise ValueError(f"decode input {past_name!r} not found for prefill output {present_name!r}.")
        if present_name not in decode_output_map:
            raise ValueError(f"decode output {present_name!r} not found.")

        prefill_shape = prefill_output_map[present_name].shape
        decode_input_shape = decode_input_map[past_name].shape
        decode_output_shape = decode_output_map[present_name].shape
        if prefill_shape != decode_input_shape:
            raise ValueError(
                f"Cache shape mismatch: {present_name} {prefill_shape} != {past_name} {decode_input_shape}."
            )
        if prefill_shape != decode_output_shape:
            raise ValueError(
                f"Decode cache output shape mismatch: {present_name} {decode_output_shape} != {prefill_shape}."
            )


def _verify_llm_lengths(part_infos, max_sequence_length, decode_sequence_length=None):
    prefill_inputs, prefill_outputs = part_infos["llm_prefill"]
    decode_inputs, _ = part_infos["llm_decode"]
    prefill_input_map = _info_by_name(prefill_inputs)
    decode_input_map = _info_by_name(decode_inputs)
    prefill_output_map = _info_by_name(prefill_outputs)

    if prefill_input_map["inputs_embeds"].shape[1] != max_sequence_length:
        raise ValueError("llm_prefill inputs_embeds sequence length does not match max_sequence_length.")
    if prefill_output_map["hidden_states"].shape[1] != max_sequence_length:
        raise ValueError("llm_prefill hidden_states sequence length does not match max_sequence_length.")
    if decode_input_map["attention_mask"].shape[-1] != max_sequence_length:
        raise ValueError("llm_decode attention_mask cache length does not match max_sequence_length.")
    if prefill_input_map["cache_position"].shape[0] != max_sequence_length:
        raise ValueError("llm_prefill cache_position length does not match max_sequence_length.")
    if decode_sequence_length is not None:
        if decode_input_map["inputs_embeds"].shape[1] != decode_sequence_length:
            raise ValueError("llm_decode inputs_embeds sequence length does not match decode_sequence_length.")
        if decode_input_map["attention_mask"].shape[2] != decode_sequence_length:
            raise ValueError("llm_decode attention_mask query length does not match decode_sequence_length.")
        if decode_input_map["cache_position"].shape[0] != decode_sequence_length:
            raise ValueError("llm_decode cache_position length does not match decode_sequence_length.")


def _verify_chain_shapes(part_infos, manifest):
    if "vit" in part_infos and "vlm" in part_infos:
        _, vit_outputs = part_infos["vit"]
        vlm_inputs, vlm_outputs = part_infos["vlm"]
        vit_output_map = _info_by_name(vit_outputs)
        vlm_input_map = _info_by_name(vlm_inputs)
        _require_info(vit_output_map, "image_embeds", "vit output")
        _require_info(vlm_input_map, "image_embeds", "vlm input")
        if vit_output_map["image_embeds"].shape != vlm_input_map["image_embeds"].shape:
            raise ValueError(
                "VIT image_embeds output shape does not match VLM image_embeds input shape: "
                f"{vit_output_map['image_embeds'].shape} != {vlm_input_map['image_embeds'].shape}."
            )

    if "embed" in part_infos and "vlm" in part_infos:
        _, embed_outputs = part_infos["embed"]
        vlm_inputs, _ = part_infos["vlm"]
        embed_output_map = _info_by_name(embed_outputs)
        vlm_input_map = _info_by_name(vlm_inputs)
        _require_info(embed_output_map, "inputs_embeds", "embed output")
        _require_info(vlm_input_map, "inputs_embeds", "vlm input")
        if embed_output_map["inputs_embeds"].shape != vlm_input_map["inputs_embeds"].shape:
            raise ValueError(
                "embed inputs_embeds output shape does not match vlm inputs_embeds input shape: "
                f"{embed_output_map['inputs_embeds'].shape} != {vlm_input_map['inputs_embeds'].shape}."
            )

    if "vlm" in part_infos and "llm_prefill" in part_infos:
        _, vlm_outputs = part_infos["vlm"]
        prefill_inputs, _ = part_infos["llm_prefill"]
        vlm_output_map = _info_by_name(vlm_outputs)
        prefill_input_map = _info_by_name(prefill_inputs)
        for name in ("position_ids", "inputs_embeds", "attention_mask", "linear_attention_mask"):
            _require_info(vlm_output_map, name, "vlm output")
            _require_info(prefill_input_map, name, "llm_prefill input")
            if vlm_output_map[name].shape != prefill_input_map[name].shape:
                raise ValueError(
                    f"VLM {name} output shape does not match llm_prefill {name} input shape: "
                    f"{vlm_output_map[name].shape} != {prefill_input_map[name].shape}."
                )

    if "llm_decode" in part_infos and "gen" in part_infos:
        _, decode_outputs = part_infos["llm_decode"]
        gen_inputs, _ = part_infos["gen"]
        decode_output_map = _info_by_name(decode_outputs)
        gen_input_map = _info_by_name(gen_inputs)
        _require_info(decode_output_map, "hidden_states", "llm_decode output")
        _require_info(gen_input_map, "hidden_states", "gen input")
        if decode_output_map["hidden_states"].shape != gen_input_map["hidden_states"].shape:
            raise ValueError(
                "llm_decode hidden_states output shape does not match gen hidden_states input shape: "
                f"{decode_output_map['hidden_states'].shape} != {gen_input_map['hidden_states'].shape}."
            )

    if "embed_select" in part_infos and "llm_decode" in part_infos:
        _, embed_select_outputs = part_infos["embed_select"]
        decode_inputs, _ = part_infos["llm_decode"]
        embed_select_output_map = _info_by_name(embed_select_outputs)
        decode_input_map = _info_by_name(decode_inputs)
        _require_info(embed_select_output_map, "inputs_embeds", "embed_select output")
        _require_info(decode_input_map, "inputs_embeds", "llm_decode input")
        if embed_select_output_map["inputs_embeds"].shape != decode_input_map["inputs_embeds"].shape:
            raise ValueError(
                "embed_select inputs_embeds output shape does not match llm_decode inputs_embeds input shape: "
                f"{embed_select_output_map['inputs_embeds'].shape} != {decode_input_map['inputs_embeds'].shape}."
            )

    if manifest is not None and "gen" in part_infos:
        decode_seq_len = _manifest_decode_sequence_length(manifest)
        if decode_seq_len is not None:
            gen_inputs, gen_outputs = part_infos["gen"]
            gen_input_map = _info_by_name(gen_inputs)
            gen_output_map = _info_by_name(gen_outputs)
            _require_info(gen_input_map, "hidden_states", "gen input")
            _require_info(gen_output_map, "logits", "gen output")
            if gen_input_map["hidden_states"].shape[1] != decode_seq_len:
                raise ValueError("gen hidden_states sequence length does not match decode_sequence_length.")
            if gen_output_map["logits"].shape[0] != gen_input_map["hidden_states"].shape[0]:
                raise ValueError("gen logits batch size does not match hidden_states batch size.")
            if gen_output_map["logits"].shape[1] != decode_seq_len:
                raise ValueError("gen logits sequence length does not match decode_sequence_length.")
            if "embed_select" in part_infos:
                embed_select_inputs, embed_select_outputs = part_infos["embed_select"]
                embed_select_input_map = _info_by_name(embed_select_inputs)
                embed_select_output_map = _info_by_name(embed_select_outputs)
                _require_info(embed_select_input_map, "cache_position", "embed_select input")
                _require_info(embed_select_output_map, "inputs_embeds", "embed_select output")
                if embed_select_input_map["cache_position"].shape[0] != decode_seq_len:
                    raise ValueError("embed_select cache_position length does not match decode_sequence_length.")
                if embed_select_output_map["inputs_embeds"].shape[1] != decode_seq_len:
                    raise ValueError("embed_select inputs_embeds sequence length does not match decode_sequence_length.")

    if manifest is not None and "vlm" in part_infos:
        image_embed_lengths = manifest.get("image_embed_lengths", [])
        vision_grid_thw = manifest.get("vision_grid_thw", [])
        spatial_merge_size = int(manifest.get("vision_spatial_merge_size", 1))
        if image_embed_lengths and vision_grid_thw:
            expected_lengths = [
                int(t) * int(h) * int(w) // (spatial_merge_size ** 2)
                for t, h, w in vision_grid_thw
            ]
            if expected_lengths != image_embed_lengths:
                raise ValueError(
                    f"manifest image_embed_lengths {image_embed_lengths} do not match "
                    f"vision_grid_thw-derived lengths {expected_lengths}."
                )
            vlm_inputs, _ = part_infos["vlm"]
            vlm_input_map = _info_by_name(vlm_inputs)
            if "image_embeds" in vlm_input_map and vlm_input_map["image_embeds"].shape[0] != sum(image_embed_lengths):
                raise ValueError("VLM image_embeds token length does not match manifest image_embed_lengths.")


def _load_manifest(onnx_dir):
    manifest_path = os.path.join(onnx_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parts_from_manifest(manifest):
    if manifest is None:
        return REQUIRED_PARTS
    parts = tuple(manifest.get("parts", {}).keys())
    if not parts:
        raise ValueError("manifest.json does not list any exported parts.")
    return parts


def _verify_manifest(manifest, max_sequence_length):
    if manifest is None:
        return max_sequence_length
    if not manifest.get("static_shape", False):
        raise ValueError("manifest.json does not mark the export as static_shape=true.")
    if not manifest.get("external_data", False):
        raise ValueError("manifest.json does not mark the export as external_data=true.")
    manifest_max_seq_len = int(manifest["max_sequence_length"])
    prompt_seq_len = int(manifest.get("prompt_sequence_length", manifest_max_seq_len))
    if prompt_seq_len < 1 or prompt_seq_len > manifest_max_seq_len:
        raise ValueError("manifest prompt_sequence_length must be in [1, max_sequence_length].")
    if max_sequence_length is not None and manifest_max_seq_len != max_sequence_length:
        raise ValueError(
            f"manifest max_sequence_length {manifest_max_seq_len} != requested {max_sequence_length}."
        )
    required_notes = (
        "vlm_inputs",
        "vlm_outputs",
        "decode_embedding",
        "external_data",
        "llm_prefill_attention_mask",
        "llm_prefill_linear_attention_mask",
        "llm_decode_attention_mask",
        "llm_decode_linear_attention_mask",
        "llm_decode_position_ids",
        "cache_position",
        "cache_handoff",
    )
    runtime_notes = manifest.get("runtime_notes", {})
    for name in required_notes:
        if name not in runtime_notes:
            raise ValueError(f"manifest runtime_notes missing {name!r}.")
    cache_mapping = manifest.get("cache", {}).get("mapping", [])
    parts = manifest.get("parts", {})
    if "llm_prefill" in parts and "llm_decode" in parts:
        present_outputs = {
            name for name in parts["llm_prefill"].get("output_shapes", {})
            if name.startswith("present_")
        }
        if not present_outputs:
            present_outputs = {
                name for name in parts["llm_prefill"].get("output_names", [])
                if name.startswith("present_")
            }
        mapped_outputs = {item.get("prefill_output") for item in cache_mapping}
        if mapped_outputs != present_outputs:
            raise ValueError(
                f"Manifest cache mapping mismatch: mapped={sorted(mapped_outputs)} present={sorted(present_outputs)}."
            )
    for item in cache_mapping:
        if not item.get("decode_input_exists", False):
            raise ValueError(f"Manifest cache mapping has missing decode input: {item}.")
        if not item.get("decode_output_exists", False):
            raise ValueError(f"Manifest cache mapping has missing decode output: {item}.")
        shape = item.get("shape")
        decode_input_shape = item.get("decode_input_shape")
        decode_output_shape = item.get("decode_output_shape")
        if _has_concrete_shape(shape) and _has_concrete_shape(decode_input_shape) and shape != decode_input_shape:
            raise ValueError(f"Manifest cache input shape mismatch: {item}.")
        if _has_concrete_shape(shape) and _has_concrete_shape(decode_output_shape) and shape != decode_output_shape:
            raise ValueError(f"Manifest cache output shape mismatch: {item}.")
    return manifest_max_seq_len



def _has_concrete_shape(shape):
    return isinstance(shape, list) and all(isinstance(dim, int) for dim in shape)


def _manifest_decode_sequence_length(manifest):
    if manifest is None:
        return None
    value = manifest.get("decode_sequence_length")
    return None if value is None else int(value)


def _verify_manifest_shapes(manifest, part_infos):
    if manifest is None:
        return
    for part, (inputs, outputs) in part_infos.items():
        part_manifest = manifest.get("parts", {}).get(part)
        if part_manifest is None:
            raise ValueError(f"manifest.json missing part {part!r}.")
        input_shapes = part_manifest.get("input_shapes", {})
        graph_input_shapes = part_manifest.get("graph_input_shapes", {})
        output_shapes = part_manifest.get("output_shapes", {})
        input_names = set(part_manifest.get("input_names", []))
        output_names = set(part_manifest.get("output_names", []))
        runtime_inputs = set(part_manifest.get("runtime_inputs", []))
        layout_inputs = set(part_manifest.get("layout_inputs", part_manifest.get("tracing_inputs", [])))
        graph_input_names = {info.name for info in inputs}
        graph_output_names = {info.name for info in outputs}

        missing_inputs = runtime_inputs - graph_input_names
        if missing_inputs:
            raise ValueError(f"{part} ONNX graph is missing manifest runtime inputs: {sorted(missing_inputs)}.")
        missing_runtime_inputs = graph_input_names - runtime_inputs
        if missing_runtime_inputs:
            raise ValueError(f"{part} manifest runtime_inputs missing graph inputs: {sorted(missing_runtime_inputs)}.")
        missing_outputs = output_names - graph_output_names
        if missing_outputs:
            raise ValueError(f"{part} ONNX graph is missing manifest outputs: {sorted(missing_outputs)}.")

        for info in inputs:
            if info.name not in input_shapes:
                raise ValueError(f"{part} input {info.name} is missing from manifest input_shapes.")
            expected = graph_input_shapes.get(info.name, input_shapes.get(info.name))
            if expected is not None and tuple(expected) != info.shape:
                raise ValueError(f"{part} input {info.name} shape {info.shape} != manifest {expected}.")
        for info in outputs:
            if info.name not in output_shapes:
                raise ValueError(f"{part} output {info.name} is missing from manifest output_shapes.")
            expected = output_shapes.get(info.name)
            if expected is not None and tuple(expected) != info.shape:
                raise ValueError(f"{part} output {info.name} shape {info.shape} != manifest {expected}.")
        for name in runtime_inputs:
            if name not in input_names:
                raise ValueError(f"{part} runtime input {name!r} is not listed in manifest input_names.")
        for name in layout_inputs:
            if name not in input_names:
                raise ValueError(f"{part} layout input {name!r} is not listed in manifest input_names.")


def _verify_runtime_inputs(manifest):
    if manifest is None:
        return
    parts = manifest.get("parts", {})
    expected_runtime_inputs = {
        "vit": ("hidden_states", "image_grid_thw"),
        "vlm": ("input_ids", "inputs_embeds", "attention_masks", "image_embeds", "mm_token_type_ids", "image_grid_thw"),
        "gen": ("hidden_states",),
        "embed": ("input_ids",),
        "embed_select": ("inputs_embeds", "cache_position"),
    }
    for part, required_inputs in expected_runtime_inputs.items():
        if part not in parts:
            continue
        runtime_inputs = set(parts[part].get("runtime_inputs", []))
        for name in required_inputs:
            if name not in runtime_inputs:
                raise ValueError(f"{part} manifest runtime_inputs missing {name!r}.")

    for part in ("llm_prefill", "llm_decode"):
        if part not in parts:
            continue
        runtime_inputs = set(parts[part].get("runtime_inputs", []))
        for name in ("position_ids", "inputs_embeds", "attention_mask", "linear_attention_mask", "cache_position"):
            if name not in runtime_inputs:
                raise ValueError(f"{part} manifest runtime_inputs missing {name!r}.")
        if not any(name.startswith("past_") for name in runtime_inputs):
            raise ValueError(f"{part} manifest runtime_inputs does not include past_* cache inputs.")


def _verify_chain_parts(parts, manifest):
    exported = set(parts)
    if ("llm_prefill" in exported) != ("llm_decode" in exported):
        raise ValueError("llm_prefill and llm_decode must be exported together for shared KV cache handoff.")
    missing = set(REQUIRED_PARTS) - exported
    complete_chain = manifest is None or manifest.get("complete_chain", not missing)
    if complete_chain and missing:
        raise ValueError(f"Exported parts do not cover the full Qwen3.5 inference chain: missing {sorted(missing)}.")


def verify_exports(onnx_dir="export/qwen35_vl_2b/ONNX", max_sequence_length=None):
    manifest = _load_manifest(onnx_dir)
    max_sequence_length = _verify_manifest(manifest, max_sequence_length)
    if max_sequence_length is None:
        raise ValueError("Pass --max-sequence-length when manifest.json is not available.")

    parts = _parts_from_manifest(manifest)
    _verify_chain_parts(parts, manifest)
    _require_parts(onnx_dir, parts)
    part_infos = _verify_static_parts(onnx_dir, parts)
    _verify_manifest_shapes(manifest, part_infos)
    _verify_runtime_inputs(manifest)
    _verify_chain_shapes(part_infos, manifest)
    if "llm_prefill" in part_infos and "llm_decode" in part_infos:
        _verify_cache_pairing(part_infos)
        _verify_llm_lengths(part_infos, max_sequence_length, _manifest_decode_sequence_length(manifest))
    print("Qwen3.5-VL ONNX export verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Verify static Qwen3.5-VL ONNX export artifacts.")
    parser.add_argument("--onnx-dir", default="export/qwen35_vl_2b/ONNX")
    parser.add_argument("--max-sequence-length", type=int, default=None)
    args = parser.parse_args()
    verify_exports(args.onnx_dir, args.max_sequence_length)


if __name__ == "__main__":
    main()
