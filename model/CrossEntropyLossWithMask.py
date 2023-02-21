from torch import nn


class CrossEntropyLossWithMask(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLossWithMask, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, prediction, target, **kwargs):
        if (mask := kwargs.get("mask", None)) is not None:
            mask = mask.bool()
            return self.base_criterion(prediction[mask], target[mask])
        return self.base_criterion(prediction, target)