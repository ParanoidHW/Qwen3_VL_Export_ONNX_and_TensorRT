import torch
from transformers import AutoProcessor


def get_model_input(config):
    processor = AutoProcessor.from_pretrained(config.qwen_path)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img_path} for img_path in config.imgs_paths]
                + [{"type": "text", "text": "Describe this image."}],
        }
    ]

    # Check context
    assert len(messages) == config.batch_size, f"messages number should be equal to batch_size, but now messages batch size = {len(messages)}, config batch_size = {config.batch_size}"

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(config.device)


def get_qwen3_onnx_input(config, torch_input, llm_hidden_size, vit_hidden_size, gen_hidden_size):
    """
    torch_input: dict contains: input_ids, pixel_values, image_grid_thw
    return: inputs of different part model
    """
    input_ids = torch_input["input_ids"].clone()    # shape torch.Size([1, 144])
    batch_size, seq_len = input_ids.shape
    deepstack_visual_len = 3

    position_ids = torch.ones(
        (3, batch_size, seq_len), dtype=torch.int64
    ).to(config.device) # torch.Size([3, 1, 144])

    inputs_embeds = torch.zeros(
        (batch_size, seq_len, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([1, 144, 2048])

    visual_pos_masks = torch.rand(batch_size, seq_len) > 0.5
    x = visual_pos_masks.sum().item()
    visual_pos_masks = visual_pos_masks.to(config.device) # torch.Size([1, 144])

    deepstack_visual_embeds = torch.randn(
        (deepstack_visual_len, x, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([3, 67, 2048])


    hidden_states = torch_input["pixel_values"].clone().to(dtype=config.dtype) # seq_len x 1536
    image_grid_thw = torch_input["image_grid_thw"].clone() # img_num x 3

    attention_masks = torch_input["attention_mask"].clone()    # shape torch.Size([1, 144])
    image_embeds = torch.randn(
        (64, vit_hidden_size), dtype=config.dtype
    )

    gen_hidden_states = torch.randn(
        (batch_size, seq_len, gen_hidden_size), dtype=config.dtype
    ).to(config.device)


    qwen3_onnx_inputs = {
        "llm": {
            "inputs": (position_ids, inputs_embeds, visual_pos_masks, deepstack_visual_embeds),
            "input_names": ["position_ids", "inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds"],
            "output_names": ["hidden_states"],
            "dynamic_axes": {
                "position_ids": {1: "batch_size", 2: "seq_length"},
                "inputs_embeds": {0: "batch_size", 1: "seq_length"},
                "visual_pos_masks": {0: "batch_size", 1: "seq_length"},
                "deepstack_visual_embeds": {1: "visual_seqlen"},
                "hidden_states": {0: "batch_size", 1: "seq_length"},
            }
        },
        "vit": {
            "inputs": (hidden_states, image_grid_thw),
            "input_names": ["hidden_states", "image_grid_thw"],
            "output_names": ["image_embeds", "deepstack_image_embeds"],
            "dynamic_axes": {
                "hidden_states": {0: "seq_len"},
                "image_grid_thw": {0: "img_num"},
            }
        },
        "vlm": {
            "inputs": (input_ids, attention_masks, image_embeds),
            "input_names": ["input_ids", "attention_masks", "image_embeds"],
            "output_names": ["position_ids", "attention_mask", "inputs_embeds", "visual_pos_masks"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "attention_masks": {0: "batch_size", 1: "seq_length"},
                "image_grid_thw": {0: "num_images"},
            },
        },
        "gen": {
            "inputs": (gen_hidden_states, ),
            "input_names": ["hidden_states"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size", 1: "seq_length"},
            },
        }
    }
    return qwen3_onnx_inputs


def get_qwen35_onnx_input(config, torch_input, llm_hidden_size, vit_hidden_size, gen_hidden_size):
    """
    torch_input: dict contains: input_ids, pixel_values, image_grid_thw
    return: inputs of different part model
    """
    _require_qwen35_input_fields(
        torch_input,
        ("input_ids", "pixel_values", "image_grid_thw", "attention_mask", "mm_token_type_ids"),
    )
    raw_input_ids = torch_input["input_ids"].clone()    # shape torch.Size([1, 144])
    batch_size, seq_len = raw_input_ids.shape
    max_seq_len = int(getattr(config, "max_sequence_length", seq_len))
    decode_seq_len = int(getattr(config, "decode_sequence_length", 1))
    if decode_seq_len < 1 or decode_seq_len > max_seq_len:
        raise ValueError(
            f"decode_sequence_length must be in [1, {max_seq_len}], got {decode_seq_len}."
        )
    if seq_len > max_seq_len:
        raise ValueError(
            f"Input sequence length {seq_len} exceeds max_sequence_length {max_seq_len}."
        )
    config.prompt_sequence_length = int(seq_len)
    input_ids = _pad_qwen35_sequence(raw_input_ids, max_seq_len, pad_value=0)

    position_ids = torch.ones(
        (4, batch_size, max_seq_len), dtype=torch.int64
    ).to(config.device) # torch.Size([4, 1, 144])

    inputs_embeds = torch.zeros(
        (batch_size, max_seq_len, llm_hidden_size), dtype=config.dtype
    ).to(config.device) # torch.Size([1, 144, 2048])

    hidden_states = torch_input["pixel_values"].clone().to(dtype=config.dtype) # seq_len x 1536
    image_grid_thw = torch_input["image_grid_thw"].clone() # img_num x 3
    image_embed_lengths = _get_qwen35_image_embed_lengths(config, image_grid_thw)
    image_embed_len = int(image_embed_lengths.sum().item())
    config.vision_grid_thw = tuple(tuple(int(value) for value in row) for row in image_grid_thw.tolist())
    config.image_embed_lengths = tuple(int(length) for length in image_embed_lengths.tolist())
    config.image_embed_length = image_embed_len

    attention_masks = _pad_qwen35_sequence(
        torch_input["attention_mask"].clone(),
        max_seq_len,
        pad_value=0,
    )
    image_embeds = torch.randn(
        (image_embed_len, vit_hidden_size), dtype=config.dtype, device=config.device
    )
    mm_token_type_ids = _pad_qwen35_sequence(
        torch_input["mm_token_type_ids"].clone(),
        max_seq_len,
        pad_value=0,
    )

    gen_hidden_states = torch.randn(
        (batch_size, decode_seq_len, gen_hidden_size), dtype=config.dtype
    ).to(config.device)
    embed_input_ids = torch.zeros(
        (batch_size, decode_seq_len), dtype=torch.int64, device=config.device
    )

    prefill_position_ids = torch.ones(
        (4, batch_size, max_seq_len), dtype=torch.int64, device=config.device
    )
    prefill_inputs_embeds = torch.zeros(
        (batch_size, max_seq_len, llm_hidden_size), dtype=config.dtype, device=config.device
    )
    causal_mask = torch.triu(
        torch.ones((batch_size, 1, max_seq_len, max_seq_len), dtype=torch.bool, device=config.device),
        diagonal=1,
    )
    padding_key_mask = (attention_masks == 0).to(device=config.device)[:, None, None, :]
    prefill_attention_mask = torch.zeros(
        (batch_size, 1, max_seq_len, max_seq_len), dtype=config.dtype, device=config.device
    )
    prefill_attention_mask = prefill_attention_mask.masked_fill(
        causal_mask | padding_key_mask,
        torch.finfo(config.dtype).min,
    )
    prefill_linear_attention_mask = attention_masks.to(dtype=config.dtype, device=config.device)
    prefill_cache_position = torch.arange(max_seq_len, dtype=torch.int64, device=config.device)

    decode_position_ids = torch.ones(
        (4, batch_size, decode_seq_len), dtype=torch.int64, device=config.device
    )
    decode_inputs_embeds = torch.zeros(
        (batch_size, decode_seq_len, llm_hidden_size), dtype=config.dtype, device=config.device
    )
    decode_attention_mask = torch.zeros(
        (batch_size, 1, decode_seq_len, max_seq_len), dtype=config.dtype, device=config.device
    )
    decode_linear_attention_mask = torch.ones(
        (batch_size, decode_seq_len), dtype=config.dtype, device=config.device
    )
    decode_start = min(seq_len, max_seq_len - decode_seq_len)
    decode_cache_position = torch.arange(
        decode_start,
        decode_start + decode_seq_len,
        dtype=torch.int64,
        device=config.device,
    )

    cache_inputs = _get_qwen35_cache_inputs(config, batch_size, max_seq_len)
    prefill_inputs = (
        prefill_position_ids,
        prefill_inputs_embeds,
        prefill_attention_mask,
        prefill_linear_attention_mask,
        prefill_cache_position,
        *cache_inputs["tensors"],
    )
    decode_inputs = (
        decode_position_ids,
        decode_inputs_embeds,
        decode_attention_mask,
        decode_linear_attention_mask,
        decode_cache_position,
        *cache_inputs["tensors"],
    )

    qwen35_onnx_inputs = {
        "llm": {
            "inputs": (position_ids, inputs_embeds),
            "input_names": ["position_ids", "inputs_embeds"],
            "output_names": ["hidden_states"],
            "dynamic_axes": {}
        },
        "vit": {
            "inputs": (hidden_states, image_grid_thw),
            "input_names": ["hidden_states", "image_grid_thw"],
            "output_names": ["image_embeds"],
            "dynamic_axes": {}
        },
        "vlm": {
            "inputs": (input_ids, attention_masks, image_embeds, mm_token_type_ids, image_grid_thw),
            "input_names": ["input_ids", "attention_masks", "image_embeds", "mm_token_type_ids", "image_grid_thw"],
            "output_names": ["position_ids", "inputs_embeds", "attention_mask", "linear_attention_mask"],
            "dynamic_axes": {},
        },
        "gen": {
            "inputs": (gen_hidden_states, ),
            "input_names": ["hidden_states"],
            "output_names": ["logits"],
            "dynamic_axes": {},
        },
        "embed": {
            "inputs": (embed_input_ids, ),
            "input_names": ["input_ids"],
            "output_names": ["inputs_embeds"],
            "dynamic_axes": {},
        },
        "llm_prefill": {
            "inputs": prefill_inputs,
            "input_names": [
                "position_ids",
                "inputs_embeds",
                "attention_mask",
                "linear_attention_mask",
                "cache_position",
                *cache_inputs["input_names"],
            ],
            "output_names": [
                "hidden_states",
                *cache_inputs["output_names"],
            ],
            "dynamic_axes": {},
        },
        "llm_decode": {
            "inputs": decode_inputs,
            "input_names": [
                "position_ids",
                "inputs_embeds",
                "attention_mask",
                "linear_attention_mask",
                "cache_position",
                *cache_inputs["input_names"],
            ],
            "output_names": [
                "hidden_states",
                *cache_inputs["output_names"],
            ],
            "dynamic_axes": {},
        },
    }
    return qwen35_onnx_inputs


def _require_qwen35_input_fields(torch_input, field_names):
    missing = [name for name in field_names if name not in torch_input]
    if missing:
        raise ValueError(
            "Qwen3.5 ONNX export input is missing required processor fields "
            f"{missing}. Use a Qwen3.5-VL processor that returns mm_token_type_ids "
            "and image_grid_thw for multimodal static-layout export."
        )


def _pad_qwen35_sequence(tensor, max_seq_len, pad_value):
    pad_len = max_seq_len - tensor.shape[1]
    if pad_len == 0:
        return tensor
    pad = torch.full(
        (tensor.shape[0], pad_len),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def _get_qwen35_image_embed_lengths(config, image_grid_thw):
    vision_config = getattr(config, "qwen_vision_config", None)
    if vision_config is None:
        raise ValueError("config.qwen_vision_config must be set before building Qwen3.5 image inputs.")
    spatial_merge_size = vision_config.spatial_merge_size
    config.vision_spatial_merge_size = int(spatial_merge_size)
    return image_grid_thw.prod(dim=1) // (spatial_merge_size ** 2)


def _get_qwen35_cache_inputs(config, batch_size, max_seq_len):
    text_config = getattr(config, "qwen_text_config", None)
    if text_config is None:
        raise ValueError("config.qwen_text_config must be set before building Qwen3.5 cache ONNX inputs.")

    tensors = []
    input_names = []
    output_names = []
    layer_types = getattr(text_config, "layer_types", ["full_attention"] * text_config.num_hidden_layers)
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            key_shape = (
                batch_size,
                text_config.num_key_value_heads,
                max_seq_len,
                head_dim,
            )
            value_shape = key_shape
            tensors.extend(
                [
                    torch.zeros(key_shape, dtype=config.dtype, device=config.device),
                    torch.zeros(value_shape, dtype=config.dtype, device=config.device),
                ]
            )
            input_names.extend([f"past_key_{layer_idx}", f"past_value_{layer_idx}"])
            output_names.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
        elif layer_type == "linear_attention":
            conv_dim = (
                text_config.linear_key_head_dim * text_config.linear_num_key_heads * 2
                + text_config.linear_value_head_dim * text_config.linear_num_value_heads
            )
            conv_shape = (
                batch_size,
                conv_dim,
                text_config.linear_conv_kernel_dim,
            )
            recurrent_shape = (
                batch_size,
                text_config.linear_num_value_heads,
                text_config.linear_key_head_dim,
                text_config.linear_value_head_dim,
            )
            tensors.extend(
                [
                    torch.zeros(conv_shape, dtype=config.dtype, device=config.device),
                    torch.zeros(recurrent_shape, dtype=config.dtype, device=config.device),
                ]
            )
            input_names.extend([f"past_conv_state_{layer_idx}", f"past_recurrent_state_{layer_idx}"])
            output_names.extend([f"present_conv_state_{layer_idx}", f"present_recurrent_state_{layer_idx}"])
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type for ONNX cache export: {layer_type}")

    return {
        "tensors": tuple(tensors),
        "input_names": input_names,
        "output_names": output_names,
    }
