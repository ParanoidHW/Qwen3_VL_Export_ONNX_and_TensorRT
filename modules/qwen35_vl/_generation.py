import torch
from torch import nn


class Qwen35VLForConditionalGenerationOpt(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = getattr(config.text_config, "vocab_size", config.vocab_size)
        self.lm_head = nn.Linear(config.text_config.hidden_size, vocab_size, bias=False)

    def forward(
            self,
            hidden_states: torch.LongTensor = None,
            **kwargs,
        ):
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits


class Qwen35VLTokenEmbeddingOpt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            **kwargs,
        ):
        return self.embed_tokens(input_ids)
