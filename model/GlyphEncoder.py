import os

import einops
import torch
from torch import nn

from model.modeling_bert import BertForMaskedLM
from utils._global import _global
from transformers import BertConfig


class GlyphEncoder(BertForMaskedLM):
    def __init__(self):
        config = _global.config
        glyph_path = config.get("model", "glyph_model")
        bert_config = BertConfig.from_pretrained(os.path.join(glyph_path, "config.json"))
        super(GlyphEncoder, self).__init__(bert_config)
        self.stru_embeddings = nn.Embedding(15, bert_config.hidden_size)
        self.keys = ["attention_mask", "position_ids", "type_ids", "char_sep_ids", "inputs_embeds"]
        self.input_keys = ["input_ids", "stru_mask", "char_mask"] + self.keys
        checkpoint = torch.load(os.path.join(glyph_path, "pytorch_model.bin"))
        self.load_state_dict(checkpoint, strict=False)
        self.use_mean_pooling = _global.config.get("meta", "use_mean_pooling", default=True)

    def forward(self, **kwargs):
        # print({key: value.shape for key, value in kwargs.items()})
        kwargs = {key: einops.rearrange(value, "b s1 s2 -> (b s1) s2") if key in self.input_keys and len(value.shape) == 3 else value for key, value in kwargs.items()}
        # print({key: value.shape for key, value in kwargs.items()})
        input_ids, stru_mask = kwargs.pop("input_ids"), kwargs.pop("stru_mask")
        stru_input_ids = input_ids * stru_mask - self.config.vocab_size
        stru_input_ids[stru_input_ids < 0] = 0
        stru_embeddings = self.stru_embeddings(stru_input_ids)
        normal_input_ids = input_ids * (1 - stru_mask)
        normal_embeddings = self.bert.embeddings.word_embeddings(normal_input_ids)
        kwargs["inputs_embeds"] = torch.einsum("bsh,bs->bsh", stru_embeddings, stru_mask) + torch.einsum("bsh,bs->bsh", normal_embeddings, 1 - stru_mask)
        inputs = {key: value for key, value in kwargs.items() if key in self.keys}
        # print({key: value.shape for key, value in inputs.items()})
        bert_output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
        # print(sequence_output.shape, char_mask.shape)
        if self.use_mean_pooling:
            char_mask = kwargs.pop("char_mask")
            char_output = torch.einsum("ijk,ij->ik", bert_output.last_hidden_state, char_mask.float())
        else:
            char_output = (hs := bert_output.hidden_states)[-1][:, 0, :] + hs[-2][:, 0, :]
        return char_output

