from metric.PRFMetric import PRFMetric


class CharDetectPRFMetric(PRFMetric):
    def __init__(self, **kwargs):
        super(CharDetectPRFMetric, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(CharDetectPRFMetric, self).__call__(*args, **kwargs, is_macro=False, is_detection=True, prefix="cd")