# from . import base
# from . import functional as F
# from .base import Activation

import re
import monai.metrics


class MonaiDice(monai.metrics.DiceMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def __name__(self):
        name = self.__class__.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(name))
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# class IoU(base.Metric):
#     __name__ = 'iou_score'
#
#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.iou(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class Fscore(base.Metric):
#
#     def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.f_score(
#             y_pr, y_gt,
#             eps=self.eps,
#             beta=self.beta,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class Accuracy(base.Metric):
#
#     def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.accuracy(
#             y_pr, y_gt,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class Recall(base.Metric):
#
#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.recall(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class Precision(base.Metric):
#
#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.precision(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )
