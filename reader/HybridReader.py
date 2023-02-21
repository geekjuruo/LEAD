import os
from collections import defaultdict

from reader.BasicReader import BasicReader
from reader.ConfusionExpandedReader import ConfusionExpandedReader
from reader.WordDictReader import WordDictReader
from utils._global import _global


class HybridReader(WordDictReader, ConfusionExpandedReader):
    def __init__(self):
        super(HybridReader, self).__init__()

    def read(self):
        confusion_file = _global.config.get("extra_dataset", "confusion_file")
        lines = open(confusion_file, "r+", encoding="utf-8").readlines()
        for line in lines:
            center, confusion = line.strip().split(":")
            center_py = self.convertor.convert_char(center)
            confusion_py = list(map(self.convertor.convert_char, confusion))
            is_similar = list(map(lambda x: 2 * len(set(x) & set(center_py)) >= len(set(center_py)), confusion_py))
            # self.confusion_set[center] = list(confusion)
            self.confusion_set[center] = list(zip(confusion, is_similar))
            if self.use_part:
                self.confusion_set[center] = [item for item in self.confusion_set[center] if item[1] is True]
        dict_file = _global.config.get("extra_dataset", "dict_file")
        dict_data = self.load_json(dict_file)
        self.dict_data = {key: ["".join([char for char in sent if "\u4e00" <= char <= "\u9fa5" or char in "，。：；？！～~｜"]) for sent in value] for key, value in dict_data.items()}
        self.dict_keys = list(self.dict_data.keys())
        self.char_word_set = defaultdict(set)
        for word in self.dict_keys:
            for char in word:
                self.char_word_set[char].add(word)
        self.similar_data = None
        if len(similar_file := _global.config.get("extra_dataset", "similar_file", default="")) > 0:
            self.similar_data = self.load_json(similar_file)
            self.similar_data = {key: {subkey: [item for item in value if item in self.dict_data] for subkey, value in self.similar_data[key].items()} for key in self.similar_data if key in self.dict_data}
        if self.use_glyph:
            self.glyph_dict = self.load_json(os.path.join(_global.config.get("model", "glyph_model"), "han_seq.json"))
            self.valid_dict_data = {key: value for key, value in self.dict_data.items() if all([char in self.glyph_dict for char in key])}
        BasicReader.read(self)

    def convert_all(self, items, name=""):
        return WordDictReader.convert_all(self, items, name)

    def convert(self, item, mode: str):
        return WordDictReader.convert(self, item, mode) | {"expanded": ConfusionExpandedReader.convert(self, item, mode)}
