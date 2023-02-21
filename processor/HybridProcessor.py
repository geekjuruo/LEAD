from processor.MultiSourceProcessor import MultiSourceProcessor
from processor.ReorderProcessor import ReorderProcessor


class HybridProcessor(MultiSourceProcessor, ReorderProcessor):
    def __init__(self):
        super(HybridProcessor, self).__init__()

    def process(self, data: list, mode: str, **kwargs):
        data_1, data_2 = [], []
        raw_inputs = [item.pop("raw_inputs") for item in data]
        for item in data:
            item_part = item["inputs"].pop("expanded")
            data_2.append({"inputs": item_part, "raw_inputs": item.pop("raw_inputs", None)})
            data_1.append(item)
        output_1 = MultiSourceProcessor.process(self, data_1, mode, **kwargs, no_raw=True)
        output_2 = ReorderProcessor.process(self, data_2, mode, **kwargs, no_raw=True)
        return {"inputs": (output_1["inputs"], output_2["inputs"]), "meta": output_2["meta"], "raw_inputs": raw_inputs}