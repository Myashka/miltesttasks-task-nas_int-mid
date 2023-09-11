import torch.nn as nn
import torch


def CS_loss(weight=None):
    """
    Returns a Cross Entropy Loss module with optional class weights.

    :param weight: Weights for each class, defaults to None
    :type weight: list, optional
    :rtype: nn.Module
    :return: Cross Entropy Loss module.
    """
    if weight:
        weight = torch.tensor(weight)
    return nn.CrossEntropyLoss(weight=weight)
