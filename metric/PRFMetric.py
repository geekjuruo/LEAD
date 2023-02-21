import torch
import transformers

from metric.BasicMetric import BasicMetric


class PRFMetric(BasicMetric):
    def __init__(self, **kwargs):
        super(PRFMetric, self).__init__(**kwargs)
        self.name = kwargs["name"]
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs["tokenizer"]
        self.removed_index = [self.tokenizer.vocab[token] for token in ["的", "地", "得"]]
        self.reset()

    def reset(self):
        self.correct_1 = self.pred_1 = self.target_1 = 0
        self.correct_all = self.total = 0

    def accumulate(self, prediction, target, inputs, mask, is_macro, is_detection, **kwargs):
        """
        :param prediction: Tensor (B, S, H)
        :param target: LongTensor (B, S)
        :param inputs: LongTensor (B, S)
        :param mask: LongTensor (B, S) \in {0,1}^{B*S}
        :param is_macro: bool
        :param is_detection: bool
        :return:
        """
        if prediction.size() != target.size():
            prediction = prediction.max(dim=-1).indices
        assert prediction.size() == target.size() == inputs.size() == mask.size()
        # if self.name == "test13":
        if self.name in ["test13", "test14"]:
            def remove_de(_source):
                source = _source.clone()
                removed_mask = torch.zeros_like(source, dtype=torch.bool)
                for index in self.removed_index:
                    removed_mask = removed_mask | (source == index)
                source[removed_mask] = inputs[removed_mask]
                return source
            prediction_removed_de = remove_de(prediction)
            correct_matrix = (prediction == target).long()
            correct_matrix_removed = (prediction_removed_de == target).long()
            diff = ((correct_matrix | correct_matrix_removed) - correct_matrix).bool()
            prediction[diff] = prediction_removed_de[diff]
            # target = remove_de(target)
        prediction_changed = (inputs != prediction)
        target_changed = (inputs != target)
        mask = mask.bool()
        if is_detection:
            correct_matrix = (prediction_changed == target_changed).long()
        else:
            correct_matrix = (prediction == target).long()
        if is_macro:
            sum_1 = lambda item: (item.bool() & mask).sum(dim=1)
            correct_sum = sum_1(correct_matrix)
            mask_sum = mask.sum(dim=1)
            pred_sum = sum_1(prediction_changed)
            target_sum = sum_1(target_changed)
            self.correct_1 += (correct_sum == mask_sum)[pred_sum > 0].sum().item()
            self.pred_1 += (pred_sum > 0).sum().item()
            self.target_1 += (target_sum > 0).sum().item()
            self.correct_all += (correct_sum == mask_sum).sum().item()
            self.total += pred_sum.shape[0]
        else:
            get_sum = lambda item, mask: item[mask].sum().item()
            self.correct_1 += get_sum(correct_matrix, mask & prediction_changed & target_changed)
            self.pred_1 += get_sum(prediction > -1, mask & prediction_changed)
            self.target_1 += get_sum(target > -1, mask & target_changed)
            self.correct_all += get_sum(correct_matrix, mask)
            self.total += mask.sum().item()

    def output(self, prefix="", **kwargs):
        metric = {"p": self.correct_1 / self.pred_1 if self.pred_1 > 0 else 1.0, "r": self.correct_1 / self.target_1 if self.target_1 > 0 else 1.0, "a": self.correct_all / self.total if self.total > 0 else 1.0}
        metric["f"] = 2 * metric["p"] * metric["r"] / (metric["p"] + metric["r"]) if metric["p"] + metric["r"] > 0 else 0.0
        return {prefix + key: value for key, value in metric.items()}

    def __call__(self, prediction, target, **kwargs):
        self.accumulate(prediction, target, **kwargs)
        return self.output(**kwargs)