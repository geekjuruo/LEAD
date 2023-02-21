import json
import time
from functools import reduce

import torch
from torch import nn

from pipeline.BasicPipeline import BasicPipeline
from utils._global import _global


class MultiTaskPipeline(BasicPipeline):
    def __init__(self):
        super(MultiTaskPipeline, self).__init__()

    def init_criterion(self):
        config = _global.config
        self.criterions = []
        for item in config.get("meta", "criterions"):
            kwargs = item.get("kwargs", {})
            try:
                criterion = _global.get_instance("model", item["class"], **kwargs)
            except Exception as e:
                criterion = getattr(nn, item["class"])(**kwargs)
            self.criterions.append((criterion, item["weight"]))
        self.metrics = {key: [_global.get_instance("metric", item, name=key, tokenizer=self.reader.tokenizer) for item in value] for key, value in config.get("metric").items()}

    def run_one_epoch(self, index: str, **kwargs) -> None:
        gradient_step = _global.config.get("train", "gradient_step")
        max_step = _global.config.get("train", "max_step")
        is_train = index.startswith("train")
        debug_step = _global.config.get("debug", "epoch_step", default=0)
        expansion_times = _global.config.get("meta", "expansion_times", default=1)
        valid_times = _global.config.get("train", "valid_times", default=0)
        part = kwargs.get("part", 0)
        part = f"-{part}" if part else ""
        if is_train:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)
        data_loader = self.data_loaders[index]
        metrics = self.metrics.get(index, [])
        for item in metrics:
            item.reset()
        start_time = time.time()
        total_loss = 0
        MA = lambda current, total, step: total * step / (step + 1) + current / (step + 1)
        metrics_result = {}
        for step, batch in enumerate(data_loader):
            if is_train and max_step > 0 and self.current_epoch * len(data_loader) + step > max_step:
                return
            meta_info = batch.pop("meta", {})
            raw_inputs = batch.pop("raw_inputs", None)
            batch = batch["inputs"]
            target = batch.get("target", None)
            kwargs = {}
            if (mask := batch.get("target_mask", None)) is not None:
                kwargs["mask"] = mask
            input_ids = batch["input_ids"]
            if self.use_swa and self.current_epoch >= self.swa_start_epoch and (not is_train):
                if self.swa_model.n_averaged == 0:
                    self.swa_model.update_parameters(self.model)
                prediction, hidden_states = self.swa_model(**batch, return_output=True)
            else:
                prediction, hidden_states = self.model(**batch, return_output=True)
            if isinstance(hidden_states, tuple):
                if len(hidden_states) == 2:
                    kwargs["hidden_states"], kwargs["all_hidden_states"] = hidden_states
                else:
                    kwargs["hidden_states"], kwargs["all_hidden_states"], kwargs["pho_hidden_states"], kwargs["res_hidden_states"] = hidden_states
            else:
                kwargs["hidden_states"] = hidden_states
            loss: torch.Tensor = 0
            expansion_flag = False
            if valid_size := meta_info.get("valid_size", False):
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
                # print(kwargs["selection"].shape, kwargs["similar"].shape, kwargs["selection"])
            for (criterion, weight) in self.criterions:
                loss += weight * criterion(prediction, target, **kwargs)
            total_loss = MA(loss.item(), total_loss, step)
            metrics_result = reduce(lambda x, y: {**x, **y}, [metric(prediction, target, inputs=input_ids, **kwargs) for metric in metrics], {})
            metrics_filter = ",".join([f"{key}={round(value, 3)}" for key, value in metrics_result.items()])
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), _global.config.get("optimizer", "max_grad_norm"))
                if (step + 1) % gradient_step == 0:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            total_time = time.time() - start_time
            print(f"\rE{self.current_epoch}{part} S{step + 1}/{len(data_loader)} {index[0].upper() + index[1:]} {int(total_time / 60)}min{round(total_time) % 60}s Loss={round(total_loss, 4)} {metrics_filter}", end="")
            if debug_step > 0 and step >= debug_step:
                break
            if is_train and (step + 1) % (size := int(len(data_loader) / valid_times + 1) + 1) == 0:
                print()
                if self.current_epoch >= self.swa_start_epoch:
                    self.swa_model.update_parameters(self.model)
                p = int((step + 1) / size)
                for key in self.data_loaders:
                    if key.startswith("test"):
                        self.run_one_epoch(key, part=p)
                self.save_checkpoint(f"{self.save_dir}/{self.current_epoch}-{p}.pkl")
                self.model.train()
                torch.set_grad_enabled(True)
        if is_train:
            self.save_checkpoint(f"{self.save_dir}/{self.current_epoch}.pkl")
            if hasattr(self.model, "save"):
                self.model.save(f"{self.save_dir}/extra")
            if hasattr(self.reader, "save"):
                self.reader.save(f"{self.save_dir}/extra")
        json.dump(metrics_result, open(f"{self.save_dir}/{self.current_epoch}{part}_{index}_metrics.json", "w+", encoding="utf-8"), indent=4)
        print()