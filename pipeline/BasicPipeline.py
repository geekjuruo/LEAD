import json
import os
import pickle
import re
import shutil
import time
from functools import reduce

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from processor.BasicProcessor import BasicProcessor
from reader.BasicReader import BasicReader
from utils._global import _global

class BasicPipeline:
    def __init__(self):
        # Variables
        self.data_loaders = {}
        self.metrics = {}
        self.current_epoch = 0
        # Constants
        self.total_epoch = _global.config.get("train", "epoch")
        self.save_dir = _global.config.get("output", "directory")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(file := os.path.join(self.save_dir, os.path.split(_global.config_file)[-1])) or not os.path.samefile(_global.config_file, file):
            shutil.copy(_global.config_file, file)
        self.optim_params_list = ["lr", "betas", "eps", "weight_decay"]
        self.no_decay = ["bias", "layernorm.weight"]
        self.use_swa = _global.config.get("meta", "use_swa", default=False)
        self.swa_start_epoch = _global.config.get("meta", "swa_start_epoch", default=0)

    def _get_optimizer(self, *models):
        return eval("_optimizer(params=_params, {})".format(", ".join([f"{key}={value}" for key, value in _global.config["optimizer"].items() if key in self.optim_params_list])), {"_optimizer": getattr(__import__("torch.optim", fromlist=["dummy"]), _global.config.get("meta", "optimizer")), "_params": [{"params": (p for model in models for name, p in model.named_parameters() if any(n in name for n in self.no_decay))}, {"params": (p for model in models for name, p in model.named_parameters() if not any(n in name for n in self.no_decay)), "weight_decay": 0.0}]})

    def _get_scheduler(self, optimizer):
        scheduler = _global.config.get("meta", "scheduler")
        if scheduler is None:
            return None
        max_step = _global.config.get("train", "max_step")
        gradient_step = _global.config.get("train", "gradient_step")
        total_step = round(self.total_epoch * len(self.data_loaders["train"]) / gradient_step)
        kwargs = {"optimizer": optimizer, "num_warmup_steps": _global.config.get("optimizer", "num_warmup_steps"), "num_training_steps": min(max_step, total_step) if max_step > 0 else total_step}
        lib = __import__("transformers.optimization", fromlist=["dummy"])
        if hasattr(lib, scheduler):
            _scheduler = getattr(lib, scheduler)(**kwargs)
        else:
            lib = __import__("torch.optim.lr_scheduler", fromlist=["dummy"])
            if hasattr(lib, scheduler):
                _scheduler = getattr(lib, scheduler)(**kwargs)
            else:
                raise NotImplementedError
        return _scheduler

    def get_optimizer(self):
        self.optimizer = self._get_optimizer(self.model)
        self.scheduler = self._get_scheduler(self.optimizer)

    def get_loader(self):
        for key, value in self.reader.datasets.items():
            shuffle = key.startswith("train")
            batch_size = _global.config.get("dataset", key, "batch_size")
            if batch_size <= 0:
                batch_size = len(value)
            self.data_loaders[key] = DataLoader(dataset=value, collate_fn=lambda data: self.processor.process(data, key), shuffle=shuffle, drop_last=False, batch_size=batch_size)

    def init_model(self):
        config = _global.config
        self.reader: BasicReader = _global.get_instance("reader", config.get("meta", "reader"))
        self.processor: BasicProcessor = _global.get_instance("processor", config.get("meta", "processor"))

        self.models = []
        for name in config.get("meta", "model"):
            model_class = _global.get_class("model", name)
            if hasattr(model_class, "from_pretrained"):
                model = model_class.from_pretrained(_global.config.get("model", "pretrained_model"))
                if hasattr(model, "after_load_pretrained"):
                    model.after_load_pretrained()
            else:
                model = model_class()
            self.models.append(model)
        # self.models = [_global.get_instance("model", name) for name in config.get("meta", "model")]
        for i, model in enumerate(self.models):
            if hasattr(model, "to"):
                self.models[i] = model.to(_global.device)
        self.model = self.models[0]
        self.swa_model = None
        if self.use_swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=lambda ax, x, num: ax + (x - ax) / (num + 1))

    def init_criterion(self):
        config = _global.config
        try:
            self.criterion = _global.get_instance("model", config.get("meta", "criterion"))
        except Exception as e:
            self.criterion = getattr(nn, config.get("meta", "criterion"))()
        self.metrics = {key: [_global.get_instance("metric", item, name=key, tokenizer=self.reader.tokenizer) for item in value] for key, value in config.get("metric").items()}

    def initialize(self):
        self.init_model()
        self.init_criterion()
        self.reader.read()
        self.get_loader()
        self.get_optimizer()

    def run_one_epoch(self, index: str, **kwargs) -> None:
        gradient_step = _global.config.get("train", "gradient_step")
        max_step = _global.config.get("train", "max_step")
        is_train = index.startswith("train")
        debug_step = _global.config.get("debug", "epoch_step", default=0)
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
            raw_inputs = batch.pop("raw_inputs", None)
            batch = batch["inputs"]
            target = batch.get("target", None)
            kwargs = {}
            if (mask := batch.get("target_mask", None)) is not None:
                kwargs["mask"] = mask
            if self.use_swa and self.current_epoch >= self.swa_start_epoch and (not is_train):
                if self.swa_model.n_averaged == 0:
                    self.swa_model.update_parameters(self.model)
                prediction = self.swa_model(**batch)
            else:
                prediction = self.model(**batch)
            loss = self.criterion(prediction, target, **kwargs)
            total_loss = MA(loss.item(), total_loss, step)
            metrics_result = reduce(lambda x, y: {**x, **y}, [metric(prediction, target, inputs=batch["input_ids"], **kwargs) for metric in metrics], {})
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
            if is_train and (step + 1) % (size := int(len(data_loader) / valid_times + 1) + 1) == 0:
                print()
                if self.use_swa and self.current_epoch >= self.swa_start_epoch:
                    self.swa_model.update_parameters(self.model)
                p = int((step + 1) / size)
                for key in self.data_loaders:
                    if key.startswith("test"):
                        self.run_one_epoch(key, part=p)
                self.save_checkpoint(f"{self.save_dir}/{self.current_epoch}-{p}.pkl")
                self.model.train()
                torch.set_grad_enabled(True)
            if debug_step > 0 and step >= debug_step:
                break
        if is_train:
            self.save_checkpoint(f"{self.save_dir}/{self.current_epoch}.pkl")
            if hasattr(self.model, "save"):
                self.model.save(f"{self.save_dir}/extra")
            if hasattr(self.reader, "save"):
                self.reader.save(f"{self.save_dir}/extra")
        json.dump(metrics_result, open(f"{self.save_dir}/{self.current_epoch}{part}_{index}_metrics.json", "w+", encoding="utf-8"), indent=4)
        print()

    def run(self):
        mode = _global.config.get("meta", "mode")
        if mode == "train":
            for self.current_epoch in range(self.total_epoch):
                for key in self.data_loaders:
                    self.run_one_epoch(key)
        elif mode == "test":
            for key in self.data_loaders:
                if key.startswith("test"):
                    self.run_one_epoch(key)
        else:
            raise NotImplementedError

    def save_checkpoint(self, ckpt_file):
        save_model = self.swa_model if self.swa_model is not None else self.model
        checkpoint = {"model": save_model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": self.current_epoch}
        pickle.dump(checkpoint, open(ckpt_file, "wb"))

    def load_checkpoint(self, ckpt_file):
        checkpoint = pickle.load(open(ckpt_file, "rb"))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["epoch"]