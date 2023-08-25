# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : loss.py
@ Time    ：2021/6/16 17:20
"""
import torch
from torch import nn
from config import config
from functools import reduce
import numpy as np
from utils.standardrange import *


class WeightedMseMae(nn.Module):
    def __init__(self, weight=config.weight):
        super().__init__()
        self.mse_weight, self.mae_weight = config.mse_mae_weights
        self.normal_loss_global_scale = config.normal_loss_global_scale
        path = config.sample
        self.area = np.load(path)["sequence_array"]
        print(
            "use localization sample in WeightedMseMae loss_function: {}, with weight of {}".format(path,
                                                                                                    config.balancing_weights))

    def forward(self, input, target, ar=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target1 = target * 255.0  # transform, restore

        sz = input.size()
        if ar is None:
            ar = self.area
        r = np.array([ar] * (sz[0] * sz[1]))
        r = r.reshape(*sz)
        r = torch.from_numpy(r).to(input.device)

        balancing_weights = config.balancing_weights
        weights = torch.ones_like(input) * balancing_weights[0]
        weights += r * (balancing_weights[1] - balancing_weights[0])
        thresholds = [mag_to_pixel(ele) for ele in config.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
      
            weights = weights + (balancing_weights[i + 2] - balancing_weights[i+1]) * (target1 >= threshold).float()
     

        mse = torch.sum(weights * ((input - target) ** 2), dim=tuple(range(1, len(sz))))
        mae = torch.sum(weights * (torch.abs((input - target))), dim=tuple(range(1, len(sz))))

        return self.normal_loss_global_scale * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))

