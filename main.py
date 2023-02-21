import argparse
import os
import random
import numpy as np
import torch

from utils._global import _global
from utils.config_parser import Config

def init_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", required=True)
    args_parser.add_argument("--variables", required=False)
    args_parser.add_argument("--checkpoint", required=False)
    args_parser.add_argument("--device", required=False)
    args_parser.add_argument("--seed", required=False, type=int)

    args = args_parser.parse_args()
    _global.config_file = args.config
    _global.config = Config(args.config)
    seed = args.seed if args.seed is not None else 19260817
    init_seeds(seed)

    def _eval(value):
        try:
            return eval(value)
        except Exception as e:
            return value

    if args.variables is not None:
        for item in args.variables.split(","):
            key, value = item.split("=")
            fields = key.split(".")
            _global.config.set(*fields, _eval(value))
    if args.device is not None:
        _global.device = args.device

    pipeline = _global.get_instance("pipeline", _global.config.get("meta", "pipeline"))
    pipeline.initialize()
    if args.checkpoint is not None:
        pipeline.load_checkpoint(args.checkpoint)
    pipeline.run()
