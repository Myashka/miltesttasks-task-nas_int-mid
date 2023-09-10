import torch

def accuracy(logits, targets):
    """
    Compute the accuracy (after softmax) of logits with respect to targets.
    
    :param logits: raw predictions from the model
    :type logits: torch.Tensor
    :param targets: ground truth labels
    :type targets: torch.Tensor
    :rtype: float
    :return: accuracy of the model's predictions
    """
    _, predicted = torch.max(logits, 1)
    correct = (predicted == targets).sum().item()
    acc = correct / targets.size(0)
    return acc
