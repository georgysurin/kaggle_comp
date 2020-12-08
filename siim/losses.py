import torch.nn as nn
import torch
from torch import einsum

logsigmoid = torch.nn.LogSigmoid()


def focal_loss(outputs, labels, gamma=2.0):
    if not (labels.size() == outputs.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(labels.size(), outputs.size()))
    max_val = (-outputs).clamp(min=0)
    loss = outputs - outputs * labels + max_val + ((-max_val).exp() + (-outputs - max_val).exp()).log()
    invprobs = logsigmoid(-outputs * (labels * 2 - 1))
    loss = (invprobs * gamma).exp() * loss
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, logits, labels):
        return focal_loss(logits, labels)


def focal_loss_2(outputs, labels, gamma=2.0):
    if not (labels.size() == outputs.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(labels.size(), outputs.size()))
    max_val = (-outputs).clamp(min=0)
    loss = outputs - outputs * labels + max_val + ((-max_val).exp() + (-outputs - max_val).exp()).log()
    invprobs = logsigmoid(-outputs * (labels * 2 - 1))
    weights = 1
    if labels.sum() == 0:
        weights = 0
    loss = (invprobs * gamma).exp() * loss * weights
    return loss.mean()


class FocalLoss_2(nn.Module):
    def __init__(self):
        super(FocalLoss_2, self).__init__()

    def forward(self, logits, labels):
        return focal_loss_2(logits, labels)


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def weighted_bce_loss_2d(outputs, labels):
    epsilon = 1e-7
    positive_coef = 10

    outputs = outputs.clamp(min=epsilon, max=(1 - epsilon))
    bce = -torch.mean(labels * torch.log(outputs) * positive_coef + (1 - labels) * torch.log(1 - outputs))

    return bce


def weighted_bce_with_logits_loss_2d(logits, labels):
    epsilon = 1e-7
    positive_coef = 50

    logits = logits.clamp(min=epsilon, max=(1 - epsilon))
    bce = labels * torch.sigmoid(torch.log(logits)) * positive_coef + (1 - labels) * torch.sigmoid(
        torch.log(1 - logits))

    return -torch.mean(bce)


def dice_loss(outputs, labels, smooth=1.0):
    outputs = flatten(outputs)
    labels = flatten(labels)

    intersection = (outputs * labels).sum()
    union = outputs.sum() + labels.sum() + smooth

    iou = (2. * intersection + smooth) / union

    return 1 - iou


def my_focal_loss(outputs, labels, gamma=2.0):
    epsilon = 1e-7
    outputs = outputs.clamp(min=epsilon, max=(1 - epsilon))
    p = outputs * labels + (1 - outputs) * (1 - labels)
    loss = - torch.pow(p, gamma) * p.log()

    return loss.mean()


def surface_loss(outputs, labels):
    multipled = einsum("bcwh,bcwh->bcwh", outputs, labels)
    loss = multipled.mean()

    return loss


class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        print('BCE DICE LOSS initialized')

    def forward(self, outputs, labels, smooth=1.0, logits=None):
        return weighted_bce_loss_2d(outputs, labels) + 2 * dice_loss(outputs, labels, smooth) \
               + my_focal_loss(outputs, labels)


class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """

    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)