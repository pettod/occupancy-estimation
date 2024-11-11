import torch


def accuracy(y_true, y_pred):
    ratio = torch.min(y_pred, y_true) / torch.max(y_pred, y_true)
    accuracy = ratio.mean().item()
    return accuracy
