# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : assesment.py
@ Time    ：2021/6/30 11:02
"""
import torch
from torch import nn
from config import config
from functools import reduce


class precision(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, threshold=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:`
        """
        target = target * 255.0  # transform, restore
        input = input * 255.0
        if threshold is None:
            threshold = config.threshold

        underThresholdT = torch.where(target < threshold, 1, 0)
        aboveThresholdT = torch.where(target >= threshold, 1, 0)

        underThresholdP = torch.where(input < threshold, 1, 0)
        aboveThresholdP = torch.where(input >= threshold, 1, 0)

        TP = torch.sum(aboveThresholdT * aboveThresholdP)
        FN = torch.sum(aboveThresholdT * underThresholdP)
        TN = torch.sum(underThresholdT * underThresholdP)
        FP = torch.sum(underThresholdT * aboveThresholdP)

        prec = TP / (TP + FP)  # 预测发生地震的点真的发生地震的概率
        return prec


class recall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, threshold=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target = target * 255.0  # transform, restore
        input = input * 255.0
        if threshold is None:
            threshold = config.threshold
        sz = target.shape

        underThresholdT = torch.where(target < threshold, 1, 0)
        aboveThresholdT = torch.where(target >= threshold, 1, 0)

        underThresholdP = torch.where(input < threshold, 1, 0)
        aboveThresholdP = torch.where(input >= threshold, 1, 0)

        TP = torch.sum(aboveThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))
        FN = torch.sum(aboveThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))
        TN = torch.sum(underThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))
        FP = torch.sum(underThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))

        rec = TP / (TP + FN)  # 发生地震的点真的被预测有地震的概率
        return torch.mean(rec)


class accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, threshold=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target = target * 255.0  # transform, restore
        input = input * 255.0
        if threshold is None:
            threshold = config.threshold
        sz = target.shape

        underThresholdT = torch.where(target < threshold, 1, 0)
        aboveThresholdT = torch.where(target >= threshold, 1, 0)

        underThresholdP = torch.where(input < threshold, 1, 0)
        aboveThresholdP = torch.where(input >= threshold, 1, 0)

        TP = torch.sum(aboveThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))
        FN = torch.sum(aboveThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))
        TN = torch.sum(underThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))
        FP = torch.sum(underThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))

        acc = (TP + TN) / (TP + TN + FN + FP)  # 预测准确
        return torch.mean(acc)


class RScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, threshold=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target = target * 255.0  # transform, restore
        input = input * 255.0
        if threshold is None:
            threshold = config.threshold
        sz = target.shape

        underThresholdT = torch.where(target < threshold, 1, 0)
        aboveThresholdT = torch.where(target >= threshold, 1, 0)

        underThresholdP = torch.where(input < threshold, 1, 0)
        aboveThresholdP = torch.where(input >= threshold, 1, 0)

        TP = torch.sum(aboveThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))  # hits
        FN = torch.sum(aboveThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))  # misses
        TN = torch.sum(underThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))
        FP = torch.sum(underThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))  # false alarms

        r = TP / (TP + FN) - FP / (TN + FP)
        return torch.mean(r)


class mcc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, threshold=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target = target * 255.0  # transform, restore
        input = input * 255.0
        if threshold is None:
            threshold = config.threshold
        sz = target.shape

        underThresholdT = torch.where(target < threshold, 1, 0)
        aboveThresholdT = torch.where(target >= threshold, 1, 0)

        underThresholdP = torch.where(input < threshold, 1, 0)
        aboveThresholdP = torch.where(input >= threshold, 1, 0)

        TP = torch.sum(aboveThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))  # hits
        FN = torch.sum(aboveThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))  # misses
        TN = torch.sum(underThresholdT * underThresholdP, dim=tuple(range(1, len(sz))))
        FP = torch.sum(underThresholdT * aboveThresholdP, dim=tuple(range(1, len(sz))))  # false alarms

        r = (TP * TN - FP * FN) / (((TP + FP) * (FN + TP) * (FN + TN) * (FP + TN))**(1/2))

        return torch.mean(r)

