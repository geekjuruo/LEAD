import yaml

class Config:
    def __init__(self, file):
        self.configs = {}
        self.file = file
        if file is not None:
            self.read(file)

    def get(self, *args, default=None):
        config = self.configs[self.file]
        while 1:
            try:
                return self._loop_call(config, "__getitem__", args)
            except Exception as e1:
                try:
                    fallback = config["meta"]["fallback"]
                    if fallback not in self.configs:
                        self.read(fallback)
                    config = self.configs[fallback]
                except Exception as e2:
                    if default is None:
                        print(f"KeyError: {args}")
                        raise e2
                    return default

    def set(self, *args):
        object = self.configs[self.file]
        for arg in args[:-2]:
            if arg not in object:
                object[arg] = {}
            object = object[arg]
        object[args[-2]] = args[-1]

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        self.configs[self.file].__setitem__(key, value)

    def _loop_call(self, object, func_name, params_list):
        next_object = getattr(object, func_name)(params_list[0])
        return self._loop_call(next_object, func_name, params_list[1:]) if len(params_list) > 1 else next_object


    def read(self, config_file):
        content = open(config_file, "r", encoding="utf-8").read()
        self.configs[config_file] = yaml.load(content, yaml.FullLoader)