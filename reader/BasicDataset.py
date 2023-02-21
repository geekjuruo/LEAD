from torch.utils.data import Dataset

from utils._global import _global


class BasicDataset(Dataset):
    def __init__(self, raw_data, converter):
        super(BasicDataset, self).__init__()
        self.raw_data = raw_data
        self.converter = converter
        self.converted_data = [None] * len(self.raw_data)
        self.no_cache = _global.config.get("meta", "no_cache", default=False)

    def __getitem__(self, item):
        if self.no_cache:
            inputs = self.converter(self.raw_data[item])
        else:
            if self.converted_data[item] is None:
                self.converted_data[item] = self.converter(self.raw_data[item])
            inputs = self.converted_data[item]
        return {"inputs": inputs, "raw_inputs": self.raw_data[item]}

    def __len__(self):
        return len(self.raw_data)