import json
import pickle

import transformers
from transformers import PreTrainedTokenizer

from reader.BasicDataset import BasicDataset
from utils._global import _global

class BasicReader:
    def __init__(self):
        # Variables
        self.datasets = {}
        self.tokenizer: PreTrainedTokenizer = transformers.BertTokenizer.from_pretrained(_global.config.get("model", "pretrained_model"))

        # Functions
        self.load_pickle = lambda file: pickle.load(open(file, "rb"))
        self.load_json = lambda file: json.load(open(file, "r+", encoding="utf-8"))
        self.tokenize = lambda text: self.tokenizer.__call__(text=text, add_special_tokens=True, max_length=512, truncation=True, return_token_type_ids=False, return_attention_mask=True)

    def read(self):
        dataset_items = _global.config.get("dataset")
        default_func = lambda file: {"json": self.load_json, "pkl": self.load_pickle}[file.split(".")[-1]](file)
        for key, values in dataset_items.items():
            # print(key, values)
            func = getattr(self, values["func"]) if "func" in values else default_func
            self.datasets[key] = BasicDataset(self.convert_all(func(values["file"]), key), lambda item, key=key: self.convert(item, mode=key))
        # exit()

    def convert_all(self, items, name=""):
        # if name == "test13":
        #     items = [{"src": item["src"], "tgt": "".join([schar if schar != tchar and tchar in ["地", "得"] else tchar for (schar, tchar) in zip(item["src"], item["tgt"])])} for item in items]
        return items

    def convert(self, item, mode: str):
        source, target = self.tokenize(item["src"]), self.tokenize(item["tgt"])
        return {"input_ids": source.input_ids, "attention_mask": source.attention_mask, "target": target.input_ids, "target_mask": [0] + source.attention_mask[1:-1] + [0]}