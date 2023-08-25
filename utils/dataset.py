# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : dataset.py
@ Time    ：2021/9/8 19:39
"""
import torch.utils.data as data
import numpy as np
import gc


class MultiDataset(data.Dataset): 
    def __init__(self, path_fa, n, dt, case_str, train, st=0):
        """
        :param path_fa:
        :param n: 
        :param dt: 
        :param train: "train" or "test" in type of str
        """
        # TODO
        # 1. Initialize file path or list of file names.
        self.st = st
        self.lenTotal = n
        self.fileList = []
        self.fileLen = []
        for i in range(0, n+st, dt):
            path_save = path_fa + str(int(i // dt))
            pathtx = path_save + case_str + train + "_x.npz"
            pathty = path_save + case_str + train + "_y.npz"
            self.fileList.append((pathtx, pathty))
            self.fileLen.append(min(i+dt, n) - i)
        self.dt = dt
        self.fileIndex = st // dt
        self.index = st
        self.indexInThisFile = st % dt
        self._x = np.load(self.fileList[self.fileIndex][0])["sequence_array"]
        self._y = np.load(self.fileList[self.fileIndex][1])["sequence_array"]

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        index = index + self.st
        _i, _i1 = divmod(index, self.dt) 
        _i, _i1 = int(_i), int(_i1)
        if _i == self.fileIndex:
            self.index = index
            self.indexInThisFile = _i1
        else:
            self.fileIndex = _i
            del self._x, self._y
            gc.collect()
            self._x = np.load(self.fileList[self.fileIndex][0])["sequence_array"]
            self._y = np.load(self.fileList[self.fileIndex][1])["sequence_array"]
            self.fileIndex, self.indexInThisFile = _i, _i1
            self.index = index
        return self._x[self.indexInThisFile], self._y[self.indexInThisFile]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.lenTotal

  


class SingleDataset(data.Dataset): 
    def __init__(self, path_fa, n, dt, case_str, train, st=0):
        """
        :param path_fa: "C:/Users/chesley/Pictures/ne7/second_half_" as default
        :param n: 15076 for train, 2542 for test
        :param dt: 1000 or 3000
        :param train: "train" or "test" in type of str
        """
        # TODO
        # 1. Initialize file path or list of file names.
        self.st = st
        self.lenTotal = n
        self.fileList = []
        self.index = st
        for i in range(0, n+st, dt):
            path_save = path_fa + str(int(i // dt))
            pathtx = path_save + case_str + train + "_x.npz"
            pathty = path_save + case_str + train + "_y.npz"
            self.fileList.append((pathtx, pathty))
        # print(self.fileList)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        self.index = index + self.st
        _x = np.load(self.fileList[self.index][0])["sequence_array"]
        _y = np.load(self.fileList[self.index][1])["sequence_array"]
        return _x[0], _y[0]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.lenTotal
