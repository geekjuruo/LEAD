from typing import Union

import pypinyin

from utils._global import _global


class PinyinConvertor:
    def __init__(self):
        self.vocab = {char: index for index, char in enumerate(["P", *[chr(c) for c in range(ord("1"), ord("5") + 1)], *[chr(c) for c in range(ord("a"), ord("z") + 1)], "U"])}
        _global.config.set("model", "pho_vocab_size", len(self.vocab))

    def convert_char(self, char):
        if len(char) != 1:
            return "U"
        p = pypinyin.pinyin(char, style=pypinyin.Style.TONE3, neutral_tone_with_five=True, errors=lambda x: ["U"])[0][0]
        p = p[-1] + p[:-1]
        return p

    def convert_string(self, string):
        pinyin_ids = [[self.vocab.get(alpha) for alpha in self.convert_char(char)] for char in string]
        return pinyin_ids, list(map(len, pinyin_ids))