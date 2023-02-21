import einops
import torch
import transformers
from torch import nn
from torch.nn import functional as F

from utils._global import _global

class WordClassifier(nn.Module):
    def __init__(self):
        super(WordClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(_global.config.get("model", "pretrained_model"))
        if _global.config.get("meta", "share_encoder", default=False):
            del self.bert
            self.bert = None
        if _global.config.get("meta", "fixed_encoder", default=False):
            self.requires_grad_(False)
        config: transformers.BertConfig = self.bert.config
        self.hidden_size = config.hidden_size
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.multi_explanation = _global.config.get("meta", "multi_explanation", default=0)
        self.use_mean_pooling = _global.config.get("meta", "use_mean_pooling", default=False)
        self.use_whitening = _global.config.get("meta", "use_whitening", default=False)
        self.covariance = self.bias = self.step = None
        if self.use_whitening:
            self.covariance = torch.zeros(config.hidden_size, config.hidden_size, device=_global.device)
            self.bias = torch.zeros(config.hidden_size, device=_global.device)
            self.step = 0
        self.use_attention = _global.config.get("meta", "use_attention", default=False)
        self.attention_transform = None
        if self.use_attention:
            self.attention_transform = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, **kwargs):
        hidden_states = kwargs["hidden_states"]
        all_hidden_states = kwargs["all_hidden_states"]
        input_attention_mask = kwargs["attention_mask"]
        # print(hidden_states.shape)
        start, end = kwargs["extra_start"], kwargs["extra_end"]
        batch_size, num_samples, _ = kwargs["extra_input_ids"].shape
        input_ids = einops.rearrange(kwargs["extra_input_ids"], "b s1 s2 -> (b s1) s2")
        attention_mask = einops.rearrange(kwargs["extra_attention_mask"], "b s1 s2 -> (b s1) s2")
        if self.multi_explanation:
            num_samples = int(num_samples / self.multi_explanation)
            index = kwargs["extra_index"]
            index_sum = index.sum().item()
            compact_input_ids = torch.zeros(index_sum, input_ids.shape[1], dtype=input_ids.dtype).to(_global.device)
            compact_attention_mask = torch.zeros_like(compact_input_ids)
            step = 0
            for i in range(index.shape[0]):
                for j in range(index.shape[1]):
                    offset = (i * num_samples + j) * self.multi_explanation
                    # print(i, j, index[i, j], step, offset)
                    # print(index.shape, input_ids.shape, compact_input_ids.shape)
                    compact_input_ids[step:step + index[i, j]] = input_ids[offset:offset + index[i, j]]
                    compact_attention_mask[step:step + index[i, j]] = attention_mask[offset:offset + index[i, j]]
                    step += index[i, j]
            input_ids = compact_input_ids
            attention_mask = compact_attention_mask
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        sequence_output = bert_output.last_hidden_state
        if self.use_whitening:
            word_hidden_states = torch.einsum("ijk,ij,i->ik", bert_output.hidden_states[-1] + bert_output.hidden_states[-2], attention_mask.float(), 1 / attention_mask.float().sum(dim=1))
            bias = (detach_hidden_states := word_hidden_states.detach()).mean(dim=0)
            covariance = torch.cov(detach_hidden_states.T)
            MA = lambda old, new: old * self.step / (self.step + 1) + new / (self.step + 1)
            self.bias, self.covariance = MA(self.bias, bias), MA(self.covariance, covariance)
            U, S, Vh = torch.linalg.svd(self.covariance)
            kernel = torch.einsum("ij,jk->ik", U, torch.diag(1 / torch.sqrt(S)))
            sequence_cls = torch.einsum("ik,kk->ik", word_hidden_states + self.bias, kernel)
        else:
            if self.use_mean_pooling or self.use_attention:
                # sequence_cls = torch.einsum("ijk,ij,i->ik", sequence_output, attention_mask.float(), 1 / attention_mask.float().sum(dim=1))
                sequence_cls = torch.einsum("ijk,ij,i->ik", bert_output.hidden_states[-1] + bert_output.hidden_states[-2], attention_mask.float(), 1 / attention_mask.float().sum(dim=1))
                # sentence_pooling = torch.einsum("ijk,ij,i->ik", hidden_states, input_attention_mask.float(), 1 / input_attention_mask.float().sum(dim=1))
                # print(all_hidden_states[-1], input_attention_mask.shape, bert_output.last_hidden_state.shape)
                sentence_pooling = torch.einsum("ijk,ij,i->ik", all_hidden_states[-1] + all_hidden_states[-2], input_attention_mask.float(), 1 / input_attention_mask.float().sum(dim=1))
            else:
                sequence_cls = sequence_output[:, 0, :]
        if self.multi_explanation:
            pooling_cls = torch.zeros(batch_size * num_samples, sequence_cls.shape[1]).to(_global.device)
            step = 0
            if self.use_mean_pooling:
                for i in range(index.shape[0]):
                    for j in range(index.shape[1]):
                        similarity = torch.einsum("j,ij,i->i", v1 := sentence_pooling[i], v2 := sequence_cls[step:step+index[i, j]], 1 / (v1.norm() * v2.norm(dim=1))).detach()
                        max_index = similarity.max(dim=0).indices
                        pooling_cls[i * num_samples + j] = v2[max_index]
                        step += index[i, j]
            elif self.use_attention:
                for i in range(index.shape[0]):
                    for j in range(index.shape[1]):
                        similarity = torch.einsum("j,ij->i", v1 := sentence_pooling[i], self.attention_transform(v2 := sequence_cls[step:step+index[i, j]])) / torch.sqrt(torch.tensor(self.hidden_size).to(_global.device))
                        pooling_cls[i * num_samples + j] = torch.einsum("i,ij->j", F.softmax(similarity, dim=0), v2)
                        step += index[i, j]
            else:
                for i in range(index.shape[0]):
                    for j in range(index.shape[1]):
                        pooling_cls[i * num_samples + j] = sequence_cls[step:step + index[i, j]].mean(dim=0)
                        step += index[i, j]
            sequence_cls = pooling_cls
        cls_output = einops.rearrange(self.transform(sequence_cls), "(b s) h -> b s h", s=num_samples)
        selected_hidden_states = torch.zeros(cls_output.shape[0], cls_output.shape[2]).to(_global.device)
        for i in range(batch_size):
            selected_hidden_states[i] = torch.mean(hidden_states[i, start[i]:end[i]], dim=0)
        selected_hidden_states = self.layer_norm(selected_hidden_states)
        selected_hidden_states = einops.repeat(selected_hidden_states, "b h -> b s h", s=cls_output.shape[1])
        score = torch.cosine_similarity(selected_hidden_states, cls_output)
        return score
