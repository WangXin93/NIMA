import torch
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F


class EMDLoss(nn.Module):
    """Earch Mover's Distance(EMD) Loss in *Neural Image Assessment*.
    """

    def __init__(self, r=2, reduction='mean'):
        super(EMDLoss, self).__init__()
        self.r = r
        self.reduction = reduction

    def forward(self, pred, target):
        cdf_pred = torch.cumsum(pred, -1)
        cdf_target = torch.cumsum(target, -1)

        samplewise_emd = (
            torch.mean(torch.abs(cdf_pred - cdf_target) ** self.r, dim=-1) ** (1 / self.r)
        )
        if self.reduction is None:
            return samplewise_emd
        elif self.reduction == 'mean':
            return torch.mean(samplewise_emd)


class AverageMeter(object):
    """Computes and stores the average and current value.

    >>> metric = AverageMeter()
    >>> metric.update(0.9)
    >>> metric.update(0.7)
    >>> metric.avg
    0.8
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelSaver(object):
    """Save the model with best specific performance."""
    def __init__(self):
        self.best = -1

    def save(self, metric, data, fname):
        if metric >= self.best:
            self.best = metric
            torch.save(data, fname)
            logging.info("Saved best model to {}".format(fname))


if __name__ == "__main__":
    pred = torch.rand(4, 5)
    target = torch.rand(4, 5)
    pred = F.normalize(pred, p=1, dim=-1)
    target = F.normalize(target, p=1, dim=-1)
    emd_loss = EMDLoss()
    print(emd_loss(pred, target))
