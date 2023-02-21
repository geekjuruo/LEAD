from metric.PRFMetric import PRFMetric


class CharCorrectPRFMetric(PRFMetric):
    def __init__(self, **kwargs):
        super(CharCorrectPRFMetric, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(CharCorrectPRFMetric, self).__call__(*args, **kwargs, is_macro=False, is_detection=False, prefix="cc")