import logging
import os
import pickle
from collections import defaultdict
from functools import reduce

import hanlp
import numpy as np
from tqdm import tqdm

from reader.BasicReader import BasicReader
from utils._global import _global


class WordDictReader(BasicReader):
    def __init__(self):
        super(WordDictReader, self).__init__()
        self.hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, devices=0, verbose=True)
        for task in list(self.hanlp_model.tasks.keys()):
            if not task.startswith("tok"):
                del self.hanlp_model[task]
        print()
        self.negative_sample = _global.config.get("extra_dataset", "negative_sample", default=0)
        self.manual_cache = {}
        self.extra_tokenize = lambda text: self.tokenizer.__call__(text=text, add_special_tokens=True, max_length=64, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_length=True)

        self.out_dir = _global.config.get("output", "directory")
        self.select_negative = _global.config.get("extra_dataset", "select_negative", default=False)
        self.multi_explanation = _global.config.get("meta", "multi_explanation", default=0)
        self.use_word_dict = _global.config.get("meta", "use_word_dict", default=True)
        self.use_glyph = _global.config.get("meta", "use_glyph", default=False)

    def read(self):
        extra_file = _global.config.get("extra_dataset", "file")
        dict_data = self.load_json(extra_file)
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
        super(WordDictReader, self).read()

    def convert_all(self, items, name=""):
        if "test" in name:
            # if name == "test13":
            #     items = [{"src": item["src"], "tgt": "".join([schar if schar != tchar and tchar in ["地", "得"] else tchar for (schar, tchar) in zip(item["src"], item["tgt"])])} for item in items]
            return items
        valid_items = []
        if os.path.exists(cache_file := f"{self.out_dir}/cache.pkl"):
            self.manual_cache = pickle.load(open(cache_file, "rb"))
            for item in tqdm(items):
                src, tgt = item["src"], item["tgt"]
                tgt_hash = str(hash(tgt)) + str(len(tgt))
                if tgt_hash in self.manual_cache:
                    valid_items.append(item)
        else:
            for item in tqdm(items):
                src, tgt = item["src"], item["tgt"]
                tgt_hash = str(hash(tgt)) + str(len(tgt))
                words = self.hanlp_model.__call__(tgt, tasks="tok/fine")["tok/fine"]

                valid_index = [i + 1 for i in range(len(words)) if words[i] in self.dict_data]
                if len(valid_index) <= 0:
                    self.hanlp_model["tok/fine"].dict_force = set(self.dict_keys)
                    words = self.hanlp_model.__call__(tgt, tasks="tok/fine")["tok/fine"]
                    valid_index = [i + 1 for i in range(len(words)) if words[i] in self.dict_data]
                    self.hanlp_model["tok/fine"].dict_force = None
                    if len(valid_index) <= 0:
                        continue
                tokens = [["CLS"], *[self.tokenizer.tokenize(word) for word in words], ["SEP"]]
                valid_items.append(item)
                word_offset = [0]
                for token in tokens:
                    word_offset.append(word_offset[-1] + len(token))
                self.manual_cache[tgt_hash] = {"words": words, "word_offset": word_offset, "valid_index": valid_index}
            pickle.dump(self.manual_cache, open(cache_file, "wb"))
        print(f"Train Size: {len(items)}, Valid Train Size: {len(valid_items)}")
        return valid_items

    def convert(self, item, mode: str):
        src, tgt = item["src"], item["tgt"]
        source, target = self.tokenize(src), self.tokenize(tgt)
        normal_output = {"input_ids": source.input_ids, "attention_mask": source.attention_mask, "target": target.input_ids, "target_mask": [0] + source.attention_mask[1:-1] + [0]}
        # if not mode.startswith("train"):
        #     return {"normal": normal_output}
        output = {"normal": normal_output}
        tgt_hash = str(hash(tgt)) + str(len(tgt))
        if (item := self.manual_cache.get(tgt_hash, None)) is not None:
            words = item["words"]
            word_offset = item["word_offset"]
            valid_index = item["valid_index"]
        else:
            words = self.hanlp_model.__call__(tgt, tasks="tok/fine")["tok/fine"]
            valid_index = [i + 1 for i in range(len(words)) if words[i] in self.dict_data]
            if len(valid_index) <= 0:
                self.hanlp_model["tok/fine"].dict_force = set(self.dict_keys)
                words = self.hanlp_model.__call__(tgt, tasks="tok/fine")["tok/fine"]
                valid_index = [i + 1 for i in range(len(words)) if words[i] in self.dict_data]
                self.hanlp_model["tok/fine"].dict_force = None
            tokens = [["CLS"], *[self.tokenizer.tokenize(word) for word in words], ["SEP"]]
            word_offset = [0]
            for item in tokens:
                word_offset.append(word_offset[-1] + len(item))
            self.manual_cache[tgt_hash] = {"words": words, "word_offset": word_offset, "valid_index": valid_index}
        error_index = [i for i in range(len(source.input_ids)) if source.input_ids[i] != target.input_ids[i]]
        np.random.shuffle(error_index)
        flag = False
        word = ""
        word_index = 10000
        start = end = -1

        for index in error_index:
            word_index = np.searchsorted(word_offset, index, 'right') - 1
            word = words[word_index - 1]
            if word in self.dict_data:
                flag = True
                break

        # if len(error_index) > 0 and not flag:
        #     index = int(np.random.choice(error_index))
        #     word = self.tokenizer.decode(target.input_ids[index])
        #     if word in self.dict_data:
        #         flag = True
        #         start = index
        #         end = index + 1

        if len(error_index) <= 0 or not flag:
            if len(valid_index) <= 0:
                print(words)
                raise RuntimeError
            word_index = np.random.choice(valid_index)
            word = words[word_index - 1]

        if start == end == -1:
            start = word_offset[word_index]
            end = word_offset[word_index + 1]

        label = np.random.choice(self.negative_sample + 1)
        extra_output = {"start": start, "end": end, "label": label}

        if self.use_word_dict:
            negative_keys = list(set(self.dict_data) - {word})
            negative_words = [negative_keys[i] for i in np.random.choice(len(negative_keys), self.negative_sample, replace=False)]
            if self.select_negative:
                if self.similar_data is not None and word in self.similar_data:
                    similar_item = self.similar_data[word]
                    similar_keys = similar_item["jin"]
                else:
                    similar_keys = list(reduce(lambda x, y: x | y, [self.char_word_set[char] for char in word], set()))
                if len(similar_keys) < self.negative_sample:
                    negative_words = similar_keys + negative_words[:self.negative_sample - len(similar_keys)]
                else:
                    negative_words = [similar_keys[i] for i in np.random.choice(len(similar_keys), self.negative_sample, replace=False)]
            sample_words = [word, *negative_words]
            sample_words[0], sample_words[label] = sample_words[label], sample_words[0]

            sample_sentences, index = [], []
            if self.multi_explanation:
                for word in sample_words:
                    sents = self.dict_data[word][:self.multi_explanation]
                    index.append(len(sents))
                    sample_sentences.extend(sents)
                    sample_sentences.extend([""] * max(0, self.multi_explanation - len(sents)))
            else:
                sample_sentences = [(sent := self.dict_data[word])[np.random.choice(len(sent))] for word in sample_words]
                # sample_sentences = [(sent := self.dict_data[word])[0 if i == label else np.random.choice(len(sent))] for i, word in enumerate(sample_words)]
                # sample_sentences = [(sent := self.dict_data[word])[0] for word in sample_words]
            sample_inputs = self.extra_tokenize(sample_sentences)
            # print(word, negative_words, label, sample_sentences, self.dict_data[word])
            # print(tokens, start, end, word_offset)
            word_extra_output = extra_output | {"input_ids": sample_inputs.input_ids, "attention_mask": sample_inputs.attention_mask, "length": max(sample_inputs.length)}
            if self.multi_explanation:
                word_extra_output = word_extra_output | {"index": tuple(index)}

            output = output | {"extra": word_extra_output}

        if self.use_glyph:
            negative_keys = list(set(self.valid_dict_data) - {word})
            negative_words = [negative_keys[i] for i in np.random.choice(len(negative_keys), self.negative_sample, replace=False)]
            sample_words = [word, *negative_words]
            sample_words[0], sample_words[label] = sample_words[label], sample_words[0]
            expand_keys = ["input_ids", "stru_mask", "attention_mask", "position_ids", "type_ids", "char_sep_ids", "char_mask"]
            sample_inputs = {key: [] for key in expand_keys}
            sample_inputs["length"] = 0
            for word in sample_words:
                sample_inputs["input_ids"].append([1])
                sample_inputs["stru_mask"].append([0])
                sample_inputs["char_sep_ids"].append([0])
                sample_inputs["position_ids"].append([0])
                sample_inputs["attention_mask"].append([1])
                sample_inputs["type_ids"].append([0])
                sample_inputs["char_mask"].append([0])
                for i, char in enumerate(word):
                    if (item := self.glyph_dict.get(char, None)) is None:
                        pass
                    _len = len(item["src_idx"])
                    sample_inputs["input_ids"][-1].extend(item["src_idx"])
                    sample_inputs["stru_mask"][-1].extend(item["type_mask"])
                    sample_inputs["attention_mask"][-1].extend([1] * _len)
                    sample_inputs["char_mask"][-1].extend([1] + [0] * (_len - 1))
                    sample_inputs["position_ids"][-1].extend(list(range(1, _len + 1)))
                    sample_inputs["char_sep_ids"][-1].extend([i + 1] * _len)
                    sample_inputs["type_ids"][-1].extend([1] + [i + 2 for i in item["type_mask"][1:]])
                sample_inputs["length"] = max(sample_inputs["length"], len(sample_inputs["input_ids"][-1]))
            output = output | {"glyph": extra_output | sample_inputs}
        return output

