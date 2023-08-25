# -*- encoding: utf-8 -*-
"""
@ Author  ：
@ File    : config.py
@ Time    ：
"""
from model import SO3ConvLstm
from modelConv import ConvLstm
from collections import OrderedDict
import torch
import numpy as np

class Config():
    def __init__(self, seq_in, seq_out):
        """
        :param seq_in:
        :param seq_out:
        """
        self.sequences_in = seq_in
        self.sequences_out = seq_out
        self.b = 960 
        self.width = 2 * self.b  
        self.height = 2 * self.b 
        self.half = False  

        self.layerAccess = ([207, 270, 393], [1526, 1649, 1712])
        self.dx = 1
        self.dy = 3  # 6  #

        self.pic = seq_in 
        self.dtday = 60
        self.percent = 0.7

        self.batch = 2 
        self.batch1 = 500
        self.batch2 = 200

        self.time_start = "1638_01" 
        self.time_end = "2021_04"
        self.path = "C:/Users/chesley/Pictures/npz/sequence_array.npz"

        self.num_workers = 0  
        self.num_epochs = 20  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare for earlystopping
        self.patience = 3

        
        self.start_mag = 4.0
        self.start_value = 100.0  
        self.target_csv = "remag_a/ce4/"


        self.sample = "/data/earthquake_zone/sample_" + str(
            self.height) + "_3" + "_collection_mag" + str(int(self.start_mag)) + ".npz"

        self.layerAccess = ([207, 270, 393], [1526, 1649, 1712])
        self.dx = 1
        self.dy = 3

        self.balancing_weights = [0.000666948429983852, 0.033, 1.0]
        self.THRESHOLDS = np.array([4]) 
        self.normal_loss_global_scale = 1 / 1875

        self.pre_parameters = (0.001, 20000, 0.7)

        self.end_pra = 8 
        self.idx = 5
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 4, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 192, 5, 3, 2]}),
                         OrderedDict({'conv4_leaky_1': [192, 192, 3, 2, 1]}),
                         OrderedDict({'s25_leaky_1': [192, 208, 8, 4]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         SO3ConvLstm(nfeature_in=208, nfeature_hidden=208, b_in=4,
                                     b_hidden=4, Sequences=self.sequences_in)
                         ]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [208, 192, 4, 2, 1]}),
                         OrderedDict({'deconv2_leaky_1': [192, 192, 4, 2, 1]}),
                         OrderedDict({'deconv3_leaky_1': [192, 192, 5, 3, 1]}),
                         OrderedDict({'deconv4_leaky_1': [192, 64, 4, 4, 0]}),
                         OrderedDict({'deconv5_relu_1': [64, 8, 7, 5, 1],
                                      'conv5_leaky_2': [8, 8, 3, 1, 1],
                                      'conv5_3': [8, 1, 1, 1, 0]
                                      })],
                        [SO3ConvLstm(nfeature_in=208, nfeature_hidden=208, b_in=4,
                                     b_hidden=4, Sequences=self.sequences_out),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
    

config = Config(seq_in=9, seq_out=1)
