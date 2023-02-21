# LEAD

Pytorch Implementation for EMNLP2022 Findings (Long Paper)

"Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking".

## Requirements

- python >= 3.9

- torch == 1.11.0

- transformers == 4.14.1

- hanlp == 2.1.0b27

- pypinyin == 0.46.0

- einops == 0.4.1

## Prepare Data and Pretrained Model

1. The raw data contains:

   - SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

   - SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

   - SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

   - Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

   You can also directly download the processed data from [ReaLiSe](https://github.com/DaDaMrX/ReaLiSe).

   Put the processed data files in the `resources/data` directory.

2. Download pretrained Chinese RoBERTa model, `chinese-roberta-wwm-ext`, from [huggingface](https://huggingface.co/hfl/chinese-roberta-wwm-ext), and put the all files in the `resources/pretrained_onlybert` directory.

3. Download the glyph-enhanced pretrained model from [GCC](https://github.com/lbe0613/GCC) and put the model files in `resources/glyph`.

## Run the Code

Run `run.sh` or directly execute the following command:

```shell
python main.py --config config/bert_phonics_dictionary_strokes.yaml
```

The `config` parameters can be replaced with other files placed in the `config` directory. And the configuration files can also be modified.

After training, run `test.sh` to test the model. The `checkpoint` parameter should be set correctly.

Trained checkpoints for SIGHAN datasets can be found [here](https://cloud.tsinghua.edu.cn/d/63b90d13487249c4b6b8/).

## Citation

```
@inproceedings{li2022learning,
  title={Learning from the Dictionary: Heterogeneous Knowledge Guided Fine-tuning for Chinese Spell Checking},
  author={Li, Yinghui and Ma, Shirong and Zhou, Qingyu and Li, Zhongli and Yangning, Li and Huang, Shulin and Liu, Ruiyang and Li, Chao and Cao, Yunbo and Zheng, Haitao},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  year={2022}
}
```

