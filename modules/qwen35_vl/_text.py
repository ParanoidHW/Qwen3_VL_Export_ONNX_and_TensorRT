import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, l2norm
import torch.nn.functional as F


_ONNX_CONV_STATE_UPDATES = {}


class _OnnxLinearCacheLayer:
    def __init__(self, conv_states=None, recurrent_states=None):
        self.conv_states = conv_states
        self.recurrent_states = recurrent_states


class _Qwen35OnnxCache:
    def __init__(self, config, cache_position, flat_cache, has_previous_state):
        self.config = config
        self.cache_position = cache_position
        self.layers = []
        self.conv_states = [None] * len(config.layer_types)
        self.recurrent_states = [None] * len(config.layer_types)
        self._full_layers = {}
        self._linear_layers = {}
        self._input_cache_tensors = []
        self._has_previous_state = has_previous_state
        flat_cache = tuple(flat_cache or ())
        expected_cache_tensors = len(config.layer_types) * 2
        if len(flat_cache) != expected_cache_tensors:
            raise ValueError(
                f"Qwen3.5 ONNX cache expects {expected_cache_tensors} flat tensors, got {len(flat_cache)}."
            )
        cache_iter = iter(flat_cache)

        for layer_idx, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                key_cache = next(cache_iter)
                value_cache = next(cache_iter)
                self._input_cache_tensors.extend([key_cache, value_cache])
                self.layers.append(None)
                self._full_layers[layer_idx] = [key_cache, value_cache]
            elif layer_type == "linear_attention":
                conv_state = next(cache_iter)
                recurrent_state = next(cache_iter)
                self._input_cache_tensors.extend([conv_state, recurrent_state])
                self.layers.append(_OnnxLinearCacheLayer(conv_state, recurrent_state))
                self.conv_states[layer_idx] = conv_state
                self.recurrent_states[layer_idx] = recurrent_state
                self._linear_layers[layer_idx] = self.layers[-1]
            else:
                raise ValueError(f"Unsupported Qwen3.5 layer type for ONNX cache export: {layer_type}")

    def has_previous_state(self, layer_idx=None):
        if layer_idx is None:
            return self._has_previous_state
        if not self._has_previous_state:
            return False
        layer_type = self.config.layer_types[layer_idx]
        if layer_type == "full_attention":
            return layer_idx in self._full_layers
        return layer_idx in self._linear_layers

    def get_seq_length(self, layer_idx=0):
        if self.cache_position is None:
            return 0
        return self.cache_position[-1] + 1

    def update(self, key_states, value_states, layer_idx):
        key_cache, value_cache = self._full_layers[layer_idx]
        cache_positions = self.cache_position.to(key_cache.device)
        key_cache = _scatter_cache(key_cache, key_states, cache_positions)
        value_cache = _scatter_cache(value_cache, value_states, cache_positions)
        self._full_layers[layer_idx] = [key_cache, value_cache]
        return key_cache, value_cache

    def update_conv_state(self, new_conv_state, layer_idx):
        self._linear_layers[layer_idx].conv_states = new_conv_state
        self.conv_states[layer_idx] = new_conv_state

    def update_recurrent_state(self, new_recurrent_state, layer_idx):
        self._linear_layers[layer_idx].recurrent_states = new_recurrent_state
        self.recurrent_states[layer_idx] = new_recurrent_state

    def to_flat_tuple(self):
        outputs = []
        input_idx = 0
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                for tensor in self._full_layers[layer_idx]:
                    outputs.append(tensor + self._input_cache_tensors[input_idx].sum() * 0)
                    input_idx += 1
            else:
                layer = self._linear_layers[layer_idx]
                conv_state = _ONNX_CONV_STATE_UPDATES.get(id(layer.conv_states), layer.conv_states)
                for tensor in (conv_state, layer.recurrent_states):
                    outputs.append(tensor + self._input_cache_tensors[input_idx].sum() * 0)
                    input_idx += 1
        return tuple(outputs)


def _scatter_cache(cache, states, cache_position):
    max_seq_len = cache.shape[2]
    positions = torch.arange(max_seq_len, device=cache.device, dtype=cache_position.dtype)
    write_mask = positions[None, :] == cache_position[:, None]
    write_mask = write_mask.to(dtype=cache.dtype).sum(dim=0).clamp(max=1)
    update = torch.zeros_like(cache)

    for token_idx in range(states.shape[2]):
        token_position = cache_position[token_idx]
        token_mask = (positions == token_position).to(dtype=cache.dtype).view(1, 1, max_seq_len, 1)
        update = update + states[:, :, token_idx:token_idx + 1, :] * token_mask

    write_mask = write_mask.view(1, 1, max_seq_len, 1)
    return cache * (1 - write_mask) + update


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_conv_state = hidden_states_new[:, :, -state_len:]
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    out = out[:, :, -seq_len:].to(hidden_states.dtype)
    _ONNX_CONV_STATE_UPDATES[id(conv_state)] = new_conv_state.to(conv_state.dtype)
    return out


class Qwen35VLTextModelOpt(Qwen3_5TextModel):
    def __init__(self, config):
        super().__init__(config)

        for decoder_layer in self.layers:
            if decoder_layer.layer_type == "linear_attention":
                decoder_layer.linear_attn.causal_conv1d_fn = None
                decoder_layer.linear_attn.causal_conv1d_update = torch_causal_conv1d_update
                decoder_layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                decoder_layer.linear_attn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

    def forward(
        self,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ):
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen35VLTextModelWithCacheOpt(Qwen35VLTextModelOpt):
    def __init__(self, config, has_previous_state):
        super().__init__(config)
        self.has_previous_state = has_previous_state

    def forward(
        self,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        linear_attention_mask: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        *past_key_values,
        **kwargs,
    ):
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        _ONNX_CONV_STATE_UPDATES.clear()
        cache = _Qwen35OnnxCache(self.config, cache_position, past_key_values, self.has_previous_state)

        for decoder_layer in self.layers:
            layer_mask = linear_attention_mask if decoder_layer.layer_type == "linear_attention" else attention_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return (hidden_states, *cache.to_flat_tuple())


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    # scale = 1 / (query.shape[-1] ** 0.5)
    scale = torch.rsqrt(torch.tensor([query.size(-1)], dtype=query.dtype, device=query.device))
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i]
        sub = attn[..., :i, :i]
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = torch.rsqrt(torch.tensor([query.size(-1)], dtype=query.dtype, device=query.device))
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
