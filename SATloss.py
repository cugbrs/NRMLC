import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropy():
    def __init__(self):
        self.crit = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, index, epoch):
        loss = self.crit(logits, targets)
        return loss


class SelfAdaptiveTrainingCE():
    def __init__(self, momentum=0.9, es=10):
        self.momentum = momentum
        self.es = es

    def __call__(self, logits, targets, epoch, weight=False):
        if epoch < self.es:
            return F.binary_cross_entropy(logits, targets)  # equal to nn.BCE
        
        # obtain prob, then update running avg
        prob = F.sigmoid(logits.detach())
        targets = self.momentum * targets + (1 - self.momentum) * prob

        # compute cross entropy loss, without reduction
        bce = F.logsigmoid(logits) * targets + F.logsigmoid(1 - logits) * (1 - targets)
        loss = torch.mean(-1 * bce, dim=1)

        # obtain weights
        weights, _ = targets.max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # sample weighted mean
        if weight:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        return loss