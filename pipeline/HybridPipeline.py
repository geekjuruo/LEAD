import json
import time
from functools import reduce

import torch

from pipeline.MultiModelPipeline import MultiModelPipeline
from utils._global import _global


class HybridPipeline(MultiModelPipeline):
    def __init__(self):
        super(HybridPipeline, self).__init__()

    def run_one_epoch(self, index: str, **kwargs) -> None:
        gradient_step = _global.config.get("train", "gradient_step")
        max_step = _global.config.get("train", "max_step")
        is_train = index.startswith("train")
        debug_step = _global.config.get("debug", "epoch_step", default=0)
        expansion_times = _global.config.get("meta", "expansion_times", default=1)
        valid_times = _global.config.get("train", "valid_times", default=0)
        part = kwargs.get("part", 0)
        part = f"-{part}" if part else ""
        # model, helper = self.models
        model = self.models[0]
        if is_train:
            # model.train()
            # helper.train()
            for _model in self.models:
                _model.train()
            torch.set_grad_enabled(True)
        else:
            # model.eval()
            # helper.eval()
            for _model in self.models:
                _model.eval()
            torch.set_grad_enabled(False)
        data_loader = self.data_loaders[index]
        metrics = self.metrics.get(index, [])
        for item in metrics:
            item.reset()
        start_time = time.time()
        total_loss = 0
        MA = lambda current, total, step: total * step / (step + 1) + current / (step + 1)
        metrics_result = {}
        output = []
        for step, batch in enumerate(data_loader):
            if is_train and max_step > 0 and self.current_epoch * len(data_loader) + step > max_step:
                return
            meta_info = batch.pop("meta", {})
            raw_inputs = batch.pop("raw_inputs", None)
            extra_batch, batch = batch["inputs"]
            target = batch.get("target", None)
            kwargs = {}
            if (mask := batch.get("target_mask", None)) is not None:
                kwargs["mask"] = mask
            input_ids = batch["input_ids"]
            if self.use_swa and self.current_epoch >= self.swa_start_epoch and (not is_train):
                if self.swa_model.n_averaged == 0:
                    self.swa_model.update_parameters(model)
                prediction, hidden_states = self.swa_model(**batch, return_output=True)
            else:
                prediction, hidden_states = model(**batch, return_output=True)

            if isinstance(hidden_states, tuple):
                if len(hidden_states) == 2:
                    kwargs["hidden_states"], kwargs["all_hidden_states"] = hidden_states
                else:
                    kwargs["hidden_states"], kwargs["all_hidden_states"], kwargs["pho_hidden_states"], kwargs["res_hidden_states"] = hidden_states
            else:
                kwargs["hidden_states"] = hidden_states
            expansion_flag = False
            if valid_size := meta_info.get("valid_size"):
                assert valid_size == extra_batch["input_ids"].shape[0], f"{input_ids.shape}\n{extra_batch['input_ids'].shape}"
                prediction = prediction[:valid_size]
                target = target[:valid_size]
                kwargs["mask"] = kwargs["mask"][:valid_size]
                kwargs["valid_size"] = valid_size
                expansion_flag = True
            elif expansion_times > 1:
                batch_size = prediction.shape[0]
                valid_size = int(batch_size / expansion_times)
                prediction = prediction[:valid_size]
                target = target[:valid_size]
                kwargs["mask"] = kwargs["mask"][:valid_size]
                expansion_flag = True
            if expansion_flag and (selection := batch.pop("selection", None)) is not None:
                kwargs["selection"] = selection[:valid_size]
                input_ids = input_ids[:valid_size]
                kwargs["similar"] = batch.pop("similar", None)[:valid_size]
            metrics_result = reduce(lambda x, y: {**x, **y}, [metric(prediction, target, inputs=input_ids, **kwargs) for metric in metrics], {})
            metrics_filter = ",".join([f"{key}={round(value, 3)}" for key, value in metrics_result.items()])
            if is_train:
                (criterion1, weight1), (criterion2, weight2) = self.criterions[:2]
                loss: torch.Tensor = weight1 * criterion1(prediction, target, **kwargs)
                loss += weight2 * criterion2(prediction, target, **kwargs)
                # reorder_index = meta_info.get("reorder_index")
                reorder_index = meta_info.get("inverted_reorder_index")
                kwargs["hidden_states"] = kwargs["hidden_states"][reorder_index]
                kwargs["all_hidden_states"] = tuple(kwargs["all_hidden_states"][i][reorder_index] for i in range(-2, 0))
                for i, (criterion, weight) in enumerate(self.criterions[2:]):
                    name = "glyph" if "glyph" in self.models[i + 1].__class__.__name__.lower() else "extra"
                    extra_label = extra_batch.pop(f"{name}_label")
                    extra_prediction = self.models[i + 1](**extra_batch, **kwargs)
                    loss += weight * criterion(extra_prediction, extra_label, **{key: value for key, value in kwargs.items() if key != "mask"})
                loss /= gradient_step
                total_loss = MA(loss.item(), total_loss, step)
                loss.backward()
                for i in range(len(self.models)):
                    torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), _global.config.get("optimizer", "max_grad_norm"))
                    if (step + 1) % gradient_step == 0:
                        self.optimizers[i].step()
                        if self.schedulers[i] is not None:
                            self.schedulers[i].step()
                        self.optimizers[i].zero_grad()
            else:
                tokenizer = self.reader.tokenizer
                prediction = prediction.max(dim=-1).indices
                prediction_tokens_list = tokenizer.batch_decode(prediction, skip_special_tokens=False)
                target_tokens_list = tokenizer.batch_decode(target, skip_special_tokens=False)
                input_tokens_list = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                for input_tokens, target_tokens, prediction_tokens in zip(input_tokens_list, target_tokens_list, prediction_tokens_list):
                    input_tokens = input_tokens.split(" ")
                    target_tokens = target_tokens.split(" ")
                    input_sep = input_tokens.index("[SEP]")
                    target_sep = target_tokens.index("[SEP]")
                    assert input_sep == target_sep
                    input_tokens = " ".join(input_tokens[1:input_sep])
                    target_tokens = " ".join(target_tokens[1:target_sep])
                    prediction_tokens = " ".join(prediction_tokens.split(" ")[1:target_sep])
                    output.append({"原文": input_tokens, "答案": target_tokens, "预测": prediction_tokens})
            total_time = time.time() - start_time
            print(f"\rE{self.current_epoch}{part} S{step + 1}/{len(data_loader)} {index[0].upper() + index[1:]} {int(total_time / 60)}min{round(total_time) % 60}s Loss={round(total_loss, 4)} {metrics_filter}", end="")
            if debug_step > 0 and step >= debug_step:
                break
            if is_train and (step + 1) % (size := int(len(data_loader) / valid_times + 1) + 1) == 0:
                print()
                if self.use_swa and self.current_epoch >= self.swa_start_epoch:
                    self.swa_model.update_parameters(model)
                p = int((step + 1) / size)
                for key in self.data_loaders:
                    if key.startswith("test"):
                        self.run_one_epoch(key, part=p)
                self.save_checkpoint(f"{self.save_dir}/{self.current_epoch}-{p}.pkl")
                # model.train()
                # helper.train()
                for _model in self.models:
                    _model.train()
                torch.set_grad_enabled(True)
        if is_train:
            self.save_checkpoint(f"{self.save_dir}/{self.current_epoch}.pkl")
        else:
            json.dump(output, open(f"{self.save_dir}/{self.current_epoch}{part}_{index}_output.json", "w+", encoding="utf-8"), ensure_ascii=False, indent=4)
        json.dump(metrics_result, open(f"{self.save_dir}/{self.current_epoch}{part}_{index}_metrics.json", "w+", encoding="utf-8"), indent=4)
        print()