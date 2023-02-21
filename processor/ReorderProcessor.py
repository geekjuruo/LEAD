from processor.BasicProcessor import BasicProcessor


class ReorderProcessor(BasicProcessor):
    def __init__(self):
        super(ReorderProcessor, self).__init__()

    def process(self, data: list, mode: str, **kwargs):
        wrap_data = [(item, index) for index, item in enumerate(data)]
        wrap_data.sort(key=lambda x: -len(x[0]["inputs"]))
        data, reorder_index = [], []
        inverted_reorder_index = [0] * len(wrap_data)
        for i, (item, index) in enumerate(wrap_data):
            reorder_index.append(index)
            inverted_reorder_index[index] = i
            data.append(item)

        length = len(data[0]["inputs"])
        flatten_data = [{"inputs": data[i]["inputs"][j], "raw_inputs": {}} for j in range(length) for i in range(len(data)) if j < len(data[i]["inputs"])]
        output = super(ReorderProcessor, self).process(flatten_data, mode, **kwargs)
        output = output | {"meta": {"valid_size": len(data), "reorder_index": reorder_index, "inverted_reorder_index": inverted_reorder_index}}
        return output