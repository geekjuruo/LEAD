import copy

import numpy as np

from processor.PinyinConvertor import PinyinConvertor
from reader.BasicReader import BasicReader
from utils._global import _global


class ConfusionExpandedReader(BasicReader):
    def __init__(self):
        super(ConfusionExpandedReader, self).__init__()
        self.positive_sample = _global.config.get("extra_dataset", "positive_sample")
        self.confusion_set = {}
        self.item_padding = _global.config.get("meta", "item_padding", default=False)
        self.convertor = PinyinConvertor()
        self.use_part = _global.config.get("meta", "use_part", default=False)

    def read(self):
        super(ConfusionExpandedReader, self).read()
        extra_file = _global.config.get("extra_dataset", "file")
        lines = open(extra_file, "r+", encoding="utf-8").readlines()
        for line in lines:
            center, confusion = line.strip().split(":")
            center_py = self.convertor.convert_char(center)
            confusion_py = list(map(self.convertor.convert_char, confusion))
            is_similar = list(map(lambda x: 2 * len(set(x) & set(center_py)) >= len(set(center_py)), confusion_py))
            # self.confusion_set[center] = list(confusion)
            self.confusion_set[center] = list(zip(confusion, is_similar))
            if self.use_part:
                self.confusion_set[center] = [item for item in self.confusion_set[center] if item[1] is True]

    def convert(self, item, mode: str):
        output = [super(ConfusionExpandedReader, self).convert(item, mode)]
        selection = -1
        similar = True
        for _ in range(self.positive_sample):
            next_output = copy.deepcopy(output[0])
            try:
                input_ids, target = next_output["input_ids"], next_output["target"]
                error_index = [i for i in range(len(input_ids)) if input_ids[i] != target[i]]
                # error_tokens = [self.tokenizer.convert_ids_to_tokens(input_ids[i]) for i in error_index]
                # error_index = [error_index[i] for i in range(len(error_index)) if error_tokens[i] in self.confusion_set]
                if len(error_index) <= 0:
                    raise NotImplementedError
                selection = int(np.random.choice(error_index))
                input_id, target_id = input_ids[selection], target[selection]
                input_token = self.tokenizer.convert_ids_to_tokens(input_id)
                target_token = self.tokenizer.convert_ids_to_tokens(target_id)
                # if input_token not in self.confusion_set or ((input_token, True) not in (confusion := self.confusion_set[input_token]) and (input_token, False) not in confusion):
                #     raise NotImplementedError
                if input_token not in self.confusion_set:
                    raise NotImplementedError
                confusion = self.confusion_set[input_token]
                candidates = list(set(confusion) - {(target_token, True), (target_token, False)})
                if len(candidates) <= 0:
                    raise NotImplementedError
                replaced_token, similar = candidates[np.random.choice(len(candidates))]
                replaced_id = self.tokenizer.convert_tokens_to_ids(replaced_token)
                next_output["input_ids"][selection] = replaced_id
                output.append(next_output)
            except NotImplementedError as e:
                if not self.item_padding:
                    break
                output.append(next_output)

        output = [item | {"selection": selection, "similar": similar} for item in output]
        return output
