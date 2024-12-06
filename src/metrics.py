import torch


def accuracy(y_pred, y_true):
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    else:
        y_pred = torch.round(y_pred)
    y_pred[y_pred < 0] = 0
    numerator = torch.min(y_pred, y_true)
    denominator = torch.max(y_pred, y_true)
    
    # Handle the case where both numerator and denominator are zero
    ratio = torch.where(
        (denominator == 0) & (numerator == 0), 
        torch.ones_like(denominator),  # Both are zero, set ratio to 1
        torch.where(denominator == 0, 
        torch.zeros_like(denominator),  # Denominator is zero, set ratio to 0
        numerator / denominator))  # Normal case: calculate ratio

    # Calculate the mean accuracy
    accuracy = ratio.mean().item()
    return accuracy


def categoricalAccuracy(y_pred, y_true):
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    else:
        y_pred = torch.round(y_pred)
    y_pred[y_pred < 0] = 0
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    return correct / total
