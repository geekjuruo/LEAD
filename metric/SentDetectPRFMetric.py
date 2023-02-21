from metric.PRFMetric import PRFMetric


class SentDetectPRFMetric(PRFMetric):
    def __init__(self, **kwargs):
        super(SentDetectPRFMetric, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(SentDetectPRFMetric, self).__call__(*args, **kwargs, is_macro=True, is_detection=True, prefix="sd")