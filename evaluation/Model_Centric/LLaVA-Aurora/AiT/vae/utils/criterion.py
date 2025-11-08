import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class MseIgnoreZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff = target[valid_mask] - pred[valid_mask]
        loss = torch.mean(torch.pow(diff, 2))

        return loss
