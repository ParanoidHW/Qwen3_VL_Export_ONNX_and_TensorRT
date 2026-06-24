import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from typing import Any, Callable, Optional, Union
import itertools


class Qwen35VLModelOpt(Qwen3_5Model):
    def __init__(self, qwen_config, onnx_config):
        self.batch_size = onnx_config.batch_size
        self.imgs_nums = len(onnx_config.imgs_paths)
        if hasattr(onnx_config, "image_embed_lengths"):
            self.image_embed_lengths = tuple(onnx_config.image_embed_lengths)
        else:
            self.image_embed_lengths = (onnx_config.image_embed_length,)
        super().__init__(qwen_config)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        grid_iter = iter(image_grid_thw)
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]

            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                elif modality_type == 1:
                    grid_thw = next(grid_iter)
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    grid_h = int(grid_thw[1].item())
                    grid_w = int(grid_thw[2].item())
                    current_pos += max(grid_h, grid_w) // spatial_merge_size
                else:
                    raise ValueError(f"Unsupported Qwen3.5 multimodal token type for ONNX export: {modality_type}")

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(int(llm_positions.max().item()) + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def get_image_features(self, image_embeds: torch.Tensor, **kwargs):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            image_embeds (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
        """
        image_embeds = torch.split(image_embeds, self.image_embed_lengths)
        return image_embeds


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids) # torch.Size([1, 144, 2048])

        # process image use vit model
        image_embeds = self.get_image_features(image_embeds)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        vision_positions, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        # self.model.rope_deltas = rope_deltas

        text_positions = attention_mask.long().cumsum(-1) - 1
        # We need this as otherwise padding tokens appear as -1 in position
        text_positions = text_positions.masked_fill(attention_mask == 0, 0)
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        batch_size, seq_len = attention_mask.shape
        causal_mask = torch.triu(
            torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=attention_mask.device),
            diagonal=1,
        )
        padding_key_mask = (attention_mask == 0)[:, None, None, :]
        llm_attention_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len), dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        llm_attention_mask = llm_attention_mask.masked_fill(
            causal_mask | padding_key_mask,
            torch.finfo(inputs_embeds.dtype).min,
        )
        linear_attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)

        return position_ids, inputs_embeds, llm_attention_mask, linear_attention_mask
