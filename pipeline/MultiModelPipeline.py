import json
import time
from functools import reduce

import torch

from pipeline.MultiTaskPipeline import MultiTaskPipeline
from utils._global import _global


class MultiModelPipeline(MultiTaskPipeline):
    def __init__(self):
        super(MultiModelPipeline, self).__init__()
        self.share_encoder = _global.config.get("meta", "share_encoder", default=False)
    
    def init_model(self):
        super(MultiModelPipeline, self).init_model()
        if self.share_encoder:
            self.models[1].bert = self.models[0].bert

    def get_optimizer(self):
        self.optimizers = []
        self.schedulers = []
        for model in self.models:
            optimizer = self._get_optimizer(model)
            scheduler = self._get_scheduler(optimizer)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        self.optimizer = self.optimizers[0]

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
        for step, batch in enumerate(data_loader):
            if is_train and max_step > 0 and self.current_epoch * len(data_loader) + step > max_step:
                return
            meta_info = batch.pop("meta", {})
            raw_inputs = batch.pop("raw_inputs", None)
            batch = batch["inputs"]
            target = batch.get("target", None)
            # print(batch.keys())
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
            metrics_result = reduce(lambda x, y: {**x, **y}, [metric(prediction, target, inputs=input_ids, **kwargs) for metric in metrics], {})
            metrics_filter = ",".join([f"{key}={round(value, 3)}" for key, value in metrics_result.items()])
            if is_train:
                criterion, weight = self.criterions[0]
                loss: torch.Tensor = weight * criterion(prediction, target, **kwargs)
                for i, (criterion, weight) in enumerate(self.criterions[1:]):
                    name = "glyph" if "glyph" in self.models[i + 1].__class__.__name__.lower() else "extra"
                    extra_label = batch.pop(f"{name}_label")
                    extra_prediction = self.models[i + 1](**batch, **kwargs)
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
            total_time = time.time() - start_time
            print(f"\rE{self.current_epoch}{part} S{step + 1}/{len(data_loader)} {index[0].upper() + index[1:]} {int(total_time / 60)}min{round(total_time) % 60}s Loss={round(total_loss, 4)} {metrics_filter}", end="")
            if debug_step > 0 and step >= debug_step:
                break
            if is_train and (step + 1) % (size := int(len(data_loader) / valid_times + 1) + 1) == 0:
                print()
                if self.current_epoch >= self.swa_start_epoch:
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
        json.dump(metrics_result, open(f"{self.save_dir}/{self.current_epoch}{part}_{index}_metrics.json", "w+", encoding="utf-8"), indent=4)
        print()