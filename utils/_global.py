import torch

class _global:
    config = None
    config_file = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    get_class = lambda attr, name: getattr(__import__("{}.{}".format(attr, name), fromlist=["dummy"]), name)
    get_instance = lambda attr, _name, *args, **kwargs: getattr(__import__("{}.{}".format(attr, _name), fromlist=["dummy"]), _name)(*args, **kwargs)
    temp_vars = {}