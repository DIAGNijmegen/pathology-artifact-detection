import re
import torch
# import warnings
# import torch.nn.functional as F
# import numpy as np
import monai.losses

from torch import nn, einsum
# from torch.autograd import Variable
# from sklearn.utils import class_weight
# from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.utils.losses import DiceLoss



def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.

    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))  # new axis order
    transposed = tensor.permute(axis_order).contiguous()  # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = transposed.view(C, -1)  # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed


class CombinedDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = DiceLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(
            x.view(1, x.size()[1], x.size()[2], x.size()[3], -1),
            y.view(1, x.size()[1], x.size()[2], x.size()[3], -1))

    @property
    def __name__(self):
        return 'combi_dice'


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


# class GeneralizedDiceLoss(BaseObject):
#     """Generalized Dice loss, second implementation variant."""
#
#     def __init__(self, activation=nn.Softmax(dim=1), smooth=1e-5):
#         super(GeneralizedDiceLoss, self).__init__()
#         self.activation = activation
#         self.smooth = smooth
#
#     def forward(self, logits, y_true):
#         probas = self.activation(logits) if self.activation else logits
#
#         input = flatten(probas)
#         target = flatten(y_true)
#         target = target.float()
#         target_sum = target.sum(-1)
#         class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)
#
#         intersect = (input * target).sum(-1) * class_weights
#         intersect = intersect.sum()
#
#         denominator = ((input + target).sum(-1) * class_weights).sum()
#         gdc = 1. - (2. * intersect / denominator.clamp(min=self.smooth))
#
#         return gdc


class GeneralizedDice(monai.losses.GeneralizedDiceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def __name__(self):
        name = self.__class__.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(name))
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
