import torch

from processor.BasicProcessor import BasicProcessor
from utils._global import _global


class MultiSourceProcessor(BasicProcessor):
    def __init__(self):
        super(MultiSourceProcessor, self).__init__()

    def process(self, data: list, mode: str, **kwargs):
        input_sample = data[0]["inputs"]
        # if "extra" in input_sample and "length" in input_sample["extra"]:
        #     extra_length = max([item["inputs"]["extra"]["length"] for item in data])
        output = {f"{key}_{subkey}": [] for key in input_sample for subkey in input_sample[key]}
        max_length = {key: max([len(item["inputs"][key]["input_ids"]) for item in data]) for key in input_sample}
        extra_length = {key: max([item["inputs"][key]["length"] for item in data]) for key in input_sample if input_sample[key].get("length", None) is not None}
        for item in data:
            for key, subitem in item["inputs"].items():
                for subkey, value in subitem.items():
                    map_key = f"{key}_{subkey}"
                    if isinstance(value, (dict, int, bool)):
                        output[map_key].append(value)
                    elif isinstance(value, list):
                        if isinstance(value[0], int):
                            output[map_key].append(value + [0] * (max_length[key] - len(value)))
                        elif isinstance(value[0], list):
                            output[map_key].append([item + [0] * (extra_length[key] - len(item)) for item in value] + [[0] * extra_length[key]] * (max_length[key] - len(value)))
                    elif isinstance(value, tuple):
                        output[map_key].append(list(value))
                    else:
                        print(item)
                        print(key, subkey, value)
                        raise NotImplementedError
        for key, value in output.items():
            if isinstance(value, list) and isinstance(value[0], (list, float, int)):
                output[key] = torch.tensor(value).to(_global.device)
        output = {key.replace("normal_", ""): value for key, value in output.items()}
        if self.use_multi_modal:
            input_ids = output["input_ids"].reshape(-1).tolist()
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            pinyin_ids, pinyin_length = self.pinyin_convertor.convert_string(input_tokens)
            output = output | {"pho_input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in pinyin_ids], batch_first=True, padding_value=0).to(_global.device), "pho_length": pinyin_length}
        if kwargs.get("no_raw", False):
            return {"inputs": output}
        return {"inputs": output, "raw_inputs": [item["raw_inputs"] for item in data]}