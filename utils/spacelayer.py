# -*- encoding: utf-8 -*-
"""
@ Author  ：
@ File    : spacelayer.py
@ Time    ：2022/9/19 10:47
"""
import numpy as np
import torch
from config import config
import matplotlib.pyplot as plt


def spacelayer(x, d=3):
    number, _0, height, width = x.shape
    x = x.reshape((-1, height, width))
    xnew = x.copy()
    dy = 2 * d  # config.dy  #
    ss, ns = config.layerAccess
    xnew = spacemini(x, ss[-1], ns[0], xnew, d, dy)
    ds = [d + _ for _ in range(1, len(ss))]
    for i, dx in enumerate(ds):
        xnew = spacemini(x, ss[-i-1], ss[-i-2], xnew, dx, dy)
        xnew = spacemini(x, ns[i], ns[i+1], xnew, dx, dy)
    # xnew[:, ns[-1]:, :] = x[:, ns[-1]:, :]
    # xnew[:, :ss[0], :] = x[:, :ss[0], :]
    #xnew = spacemini(x, 0, ss[0], xnew, 3, 6)
    #xnew = spacemini(x, ns[-1], 3839, xnew, 3, 6)

    xnew = xnew.reshape((number, _0, height, width))
    return xnew
