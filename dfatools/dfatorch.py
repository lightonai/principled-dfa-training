""" Tools for implementing Direct Feedback Alignment (DFA) in PyTorch.
"""
import copy
import numpy as np
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from collections import OrderedDict


class AsymmetricSequential(nn.Sequential):
    """A PyTorch sequential container modified to allow for asymmetric feedbacks.

    Extends PyTorch :class:`nn.Sequential` to allow for manual error vectors update in
    :class:`AssymetricFeedback` modules and automatic registration of feedback hooks --
    thus enabling asymmetric behaviors."""
    def __init__(self, rp_device, *args):
        super(AsymmetricSequential, self).__init__(*args)
        self.entry_points = OrderedDict()  # will store entry points to compute local gradients
        self.feedbacks = OrderedDict()  # will store dim2 of feedback matrices
        self.random_matrix = None  # will store the biggest random matrix
        self.input_shape = None
        self.rp_device = rp_device
        self.asym_locations = {}
        self.feedback_normalization = True

    def build_feedback(self, input):
        self.input_shape = input.shape
        entry_point_counter = 0
        for module in self._modules.values():
            if isinstance(module, AsymmetricFeedback):
                # store the entry point to the graph,
                # then the ErrorFeedback will detach
                # the input to the next module
                entry_point_counter += 1
                dim2 = np.prod(input.shape[1:])
                self.feedbacks['ep{}'.format(entry_point_counter)] = dim2

            input = module(input)

        classes = np.prod(input.shape[1:])  # shape at the output - if not 2D, takes product of all dims except first
        dim2 = max(self.feedbacks.values())
        if self.feedback_normalization:
            self.random_matrix = torch.randn(classes, dim2) / np.sqrt(dim2 * classes)
        else:
            self.random_matrix = torch.randn(classes, dim2)
        self.random_matrix = self.random_matrix.to(self.rp_device)
        return input

    def forward(self, input):
        if self.input_shape is None:
            raise Exception('Before calling `forward` for the first time, call `build_feedback`.')

        entry_point_counter = 0
        for module in self._modules.values():
            if isinstance(module, AsymmetricFeedback):
                # store the entry point to the graph,
                # then the ErrorFeedback will detach
                # the input to the next module
                entry_point_counter += 1
                self.entry_points['ep{}'.format(entry_point_counter)] = input
                self.asym_locations['ep{}'.format(entry_point_counter)] = module
            input = module(input)
        return input

    def backward(self, error):
        error = error.to(self.rp_device)
        self.rp = torch.mm(error, self.random_matrix)
        dim_max = max(self.feedbacks.values())
        for i, key in enumerate(self.entry_points.keys()):
            entry_point = self.entry_points[key]
            dim2 = self.feedbacks[key]
            rp_local = self.rp[:entry_point.shape[0], :dim2]
            rp_local = rp_local.view(*entry_point.shape)
            if self.feedback_normalization:
                rp_local = rp_local * np.sqrt(dim_max / dim2)
            self.asym_locations[key].rp = rp_local
            rp_local = rp_local.to(entry_point.device)
            entry_point.backward(rp_local)
        return


class AsymmetricFeedback(nn.Module):
    """PyTorch module for asymmetric feedbacks.

    The forward pass is left untouched, but will detach local parts of the graph to prevent gradients from flowing.
    """
    def __init__(self):
        super(AsymmetricFeedback, self).__init__()
        self.rp = None

    def forward(self, input):
        """Dummy forward method. """
        output = input.detach()
        output.requires_grad = True
        return output


def build_dfa_from_bp(bp_model):
    """Builds a DFA-enabled PyTorch model description from a classic backpropagation model description.

    It takes the :class:`OrderedDict` needed by PyTorch :class:`nn.Sequential` to create the network and
    adds the necessary :class:`AssymetricFeedback` modules. The new :class:`OrderedDict` can be made into a DFA-enabled
    PyTorch model by using :class:`AsymmetricSequential`.

    :param bp_model: model description
    :type bp_model: OrderedDict[str, torch.nn.Module]
    :return: DFA-enabled model description
    :rtype: OrderedDict[str, torch.nn.Module]
    """
    dfa_model = OrderedDict()
    layer_index, input_layer = 0, True
    last_layer = None
    for layer_name, layer in bp_model.items():
        if (type(layer) == torch.nn.Linear or type(layer) == nn.Conv2d) \
           and layer_index != 0 and layer_index != len(bp_model) and not input_layer:
            # Presence of a linear module marks the beginning of a new layer (except for the first and last one).
            # Before starting it, we complete the previous layer with a module to enable DFA.
            dfa_model['ef{0}'.format(str(int(layer_name[-1]) - 1))] = AsymmetricFeedback()
        dfa_model[layer_name] = copy.deepcopy(layer)
        layer_index += 1
        if layer_name != 'flat' and type(layer) != nn.Dropout and type(layer) != nn.Dropout2d and type(layer) != nn.Dropout3d:
            input_layer = False
        last_layer = layer_name

    return dfa_model
