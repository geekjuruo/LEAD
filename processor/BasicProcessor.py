import torch
import transformers
from transformers import PreTrainedTokenizer

from processor.PinyinConvertor import PinyinConvertor
from utils._global import _global


class BasicProcessor:
    def __init__(self):
        self.use_multi_modal = _global.config.get("meta", "use_multi_modal", default=False)
        self.tokenizer: PreTrainedTokenizer = transformers.BertTokenizer.from_pretrained(_global.config.get("model", "pretrained_model"))
        self.pinyin_convertor = None
        if self.use_multi_modal:
            self.pinyin_convertor = PinyinConvertor()

    def process(self, data: list, mode: str, **kwargs):
        output = {key: [] for key in data[0]["inputs"]}
        max_length = max([len(item["inputs"]["input_ids"]) for item in data])
        for item in data:
            for key, value in item["inputs"].items():
                if isinstance(value, (dict, int)):
                    output[key].append(value)
                elif isinstance(value, list) and isinstance(value[0], int):
                    output[key].append(value + [0] * (max_length - len(value)))
                else:
                    print(key, value, type(value))
                    raise NotImplementedError
        for key, value in output.items():
            if isinstance(value, list) and isinstance(value[0], (list, float, int)):
                output[key] = torch.tensor(value).to(_global.device)
        if self.use_multi_modal:
            input_ids = output["input_ids"].reshape(-1).tolist()
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            pinyin_ids, pinyin_length = self.pinyin_convertor.convert_string(input_tokens)
            output = output | {"pho_input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in pinyin_ids], batch_first=True, padding_value=0).to(_global.device), "pho_length": pinyin_length}
        if kwargs.get("no_raw", False):
           return {"inputs": output}
        return {"inputs": output, "raw_inputs": [item["raw_inputs"] for item in data]}