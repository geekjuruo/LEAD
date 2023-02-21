from metric.PRFMetric import PRFMetric


class SentCorrectPRFMetric(PRFMetric):
    def __init__(self, **kwargs):
        super(SentCorrectPRFMetric, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(SentCorrectPRFMetric, self).__call__(*args, **kwargs, is_macro=True, is_detection=False, prefix="sc")