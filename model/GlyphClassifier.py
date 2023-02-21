import einops
import torch
import transformers
from torch import nn

from model.GlyphEncoder import GlyphEncoder
from utils._global import _global


class GlyphClassifier(nn.Module):
    def __init__(self):
        super(GlyphClassifier, self).__init__()
        self.bert = GlyphEncoder()
        self.bert.requires_grad_(False)
        config: transformers.PretrainedConfig = self.bert.config
        self.hidden_size = config.hidden_size
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, **kwargs):
        hidden_states = kwargs["hidden_states"]
        start, end = kwargs["glyph_start"], kwargs["glyph_end"]
        batch_size, num_samples, _ = kwargs["glyph_input_ids"].shape
        glyph_inputs = {key.replace("glyph_", ""): value for key, value in kwargs.items() if key.startswith("glyph_")}
        glyph_output = self.bert(**glyph_inputs)
        glyph_output = einops.rearrange(self.transform(glyph_output), "(b s) h -> b s h", s=num_samples)
        selected_hidden_states = torch.zeros(glyph_output.shape[0], glyph_output.shape[2]).to(_global.device)
        for i in range(batch_size):
            selected_hidden_states[i] = torch.mean(hidden_states[i, start[i]:end[i]], dim=0)
        selected_hidden_states = einops.repeat(self.layer_norm(selected_hidden_states), "b h -> b s h", s=glyph_output.shape[1])
        # print(glyph_output.shape, selected_hidden_states.shape)
        score = torch.cosine_similarity(selected_hidden_states, glyph_output)
        return score