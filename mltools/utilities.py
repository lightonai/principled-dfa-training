""" Various functions and oddities to make life easier.
"""

import enum
import torch
import torch.nn as nn


class OrderedEnum(enum.Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Flatten(nn.Module):
    """PyTorch module for flattening 2D data to 1D.

    Straight-up application of :func:`torch.Tensor.view`, but easier to use in an :class:`OrderedDict`.
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


def to_categorical(y, num_classes):
    """Converts a class vector made of integers to a categorical matrix.

    :param y: class vector of integers
    :type y: torch.Tensor
    :param num_classes: number of classes in y
    :type num_classes: int
    :rtype: torch.Tensor
    """
    categorical = torch.eye(num_classes)[y]
    return categorical


WEIGHT_MODULES = (nn.Linear, nn.Conv2d)
