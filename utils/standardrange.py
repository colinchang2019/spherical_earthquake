# -*- encoding: utf-8 -*-
"""
@ Author  ：
@ File    : standardrange.py
@ Time    ：2021/8/25 19:43
"""
import numpy as np
from config import config


def logTomag(df, st_va=config.start_value):
    lgengmin, lgengmax = config.start_mag, 10  
    lgLgNewmin, lgLgNewmax = np.log10(0.5), np.log10(lgengmax - (config.start_mag - 0.5))

    res = (df - st_va) / (255.0 - st_va) * (lgLgNewmax - lgLgNewmin) + lgLgNewmin 
    res = np.power(10, res) + (config.start_mag - 0.5)  # MagNew

    res = np.where(df >= st_va, res, 0)
    return res

def mag_to_pixel(x, st_va=config.start_value):
    if x < config.start_mag:
        return 0
    lgengmin, lgengmax = config.start_mag, 10  
    lgLgNewmin, lgLgNewmax = np.log10(0.5), np.log10(lgengmax - (config.start_mag - 0.5))
    x = np.log10(x - (config.start_mag - 0.5))
    x = (x - lgLgNewmin) / (lgLgNewmax - lgLgNewmin) * (255.0 - st_va) + st_va
    return x


def pixel_to_mag(x, st_va=config.start_value, _np=False):
    if not _np:
        if x < st_va:
            return 0
    lgengmin, lgengmax = config.start_mag, 10 
    lgLgNewmin, lgLgNewmax = np.log10(0.5), np.log10(lgengmax - (config.start_mag - 0.5))
    x = (x - st_va) / (255.0 - st_va) * (lgLgNewmax - lgLgNewmin) + lgLgNewmin
    x = np.power(10, x) + (config.start_mag - 0.5)

    if _np:
        x = np.where(x >= 0, x, 0)

    return x
