import random

import torch
from torch import nn
import einops

from utils._global import _global


class InfoNCELossForCSC(nn.Module):
    def __init__(self, **kwargs):
        super(InfoNCELossForCSC, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss()
        self.negative_sample = kwargs["negative_sample"]
        self.temperature = kwargs["temperature"]
        self.positive_sample = _global.config.get("extra_dataset", "positive_sample", default=1)
        self.item_padding = _global.config.get("meta", "item_padding", default=False)
        self.normalized = kwargs.get("normalized", False)
        self.multi_modal = kwargs.get("multi_modal", False)

    def forward(self, *args, **kwargs):
        valid_size = kwargs["valid_size"]
        _selection = kwargs["selection"]
        hidden_states = kwargs["hidden_states"]
        hidden_states_list = [hidden_states]
        if self.multi_modal:
            hidden_states_list += [kwargs["pho_hidden_states"], kwargs["res_hidden_states"]]
        mask = kwargs["mask"]
        length = mask.sum(dim=-1)
        loss_size = hidden_states.shape[0] - valid_size
        if loss_size <= 0:
            return 0
        if self.item_padding:
            assert valid_size == int(loss_size / self.positive_sample)
            loss_size = int(loss_size / self.positive_sample)
        loss = 0
        for k in range(self.positive_sample):
            for l, _hidden_states in enumerate(hidden_states_list):
                if l == 0:
                    continue
                loss_size = int((_hidden_states.shape[0] - kwargs["valid_size"]) / self.positive_sample)
                similar = kwargs["similar"][:loss_size]
                # print("\nfuck:", l, _hidden_states.shape[0], loss_size)
                _cat = lambda hs, inc: torch.cat((hs[:loss_size][inc], hs[kwargs["valid_size"]:][inc]), dim=0)
                _cut = lambda hs, inc: hs[:loss_size][inc]
                hidden_states = _hidden_states if l == 0 else (_cat(_hidden_states, similar) if l == 1 else _cat(_hidden_states, ~similar))
                if hidden_states.shape[0] <= 0:
                    continue
                selection = _selection if l == 0 else (_cut(_selection, similar) if l == 1 else _cut(_selection, ~similar))
                loss_size = loss_size if l == 0 else (similar.sum().item() if l == 1 else (~similar).sum().item())
                valid_size = kwargs["valid_size"] if l == 0 else loss_size
                # print(l, hidden_states.shape[0], valid_size, loss_size)
                anchor = torch.zeros(hidden_states.shape[0], hidden_states.shape[2], device=hidden_states.device)
                positive = torch.zeros(hidden_states.shape[0], hidden_states.shape[2], device=hidden_states.device)
                negative = torch.zeros(hidden_states.shape[0], self.negative_sample, hidden_states.shape[2], device=hidden_states.device)
                for i in range(loss_size):
                    anchor[i] = hidden_states[i, selection[i]]
                    if self.item_padding:
                        positive[i] = hidden_states[valid_size + k * loss_size + i, selection[i]]
                    else:
                        positive[i] = hidden_states[valid_size + i, selection[i]]
                    for j in range(self.negative_sample):
                        sample = random.randint(1, length[i].item() - 1)
                        while sample == selection[i].item():
                            sample = random.randint(1, length[i].item() - 1)
                        negative[i, j] = hidden_states[i, sample]
                positive_logits = torch.einsum("bh,bh->b", anchor, positive)
                negative_logits = torch.einsum("bh,bsh->bs", anchor, negative)
                if self.normalized:
                    eps = 1e-8
                    positive_logits = positive_logits / ((anchor_norm := anchor.norm(dim=1)) * positive.norm(dim=1) + eps)
                    negative_logits = negative_logits / (torch.einsum("b,bs->bs", anchor_norm, negative.norm(dim=2)) + eps)
                labels = torch.zeros(hidden_states.shape[0], dtype=torch.long, device=hidden_states.device)
                logits = torch.cat([einops.rearrange(positive_logits, "(b s) -> b s", s=1), negative_logits], dim=1)
                loss += self.base_criterion(logits / self.temperature, labels)
        return loss / self.positive_sample
