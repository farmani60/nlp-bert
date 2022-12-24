import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    # we use binary cross-entropy with logits which first
    # applies sigmoid and then calculates the loss
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
