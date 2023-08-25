# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : 
@ Time    ：
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
from torch.autograd import Variable

from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import s2_equatorial_grid, so3_equatorial_grid, SO3Convolution, so3_integrate


class SO3ConvLSTMCell(nn.Module):
    def __init__(self, nfeature_in=16, nfeature_hidden=16, b_in=64, b_hidden=64):
        super(SO3ConvLSTMCell, self).__init__()

        self.nfeature_in = nfeature_in
        self.nfeature_hidden = nfeature_hidden
        self.b_in = b_in
        self.b_hidden = b_hidden
        # self.grid_s2 = s2_equatorial_grid(max_beta=0, n_alpha=2*b_hidden, n_beta=1)   # reckoning
        self.grid_so3 = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_hidden, n_beta=1, n_gamma=1)

        self.Wxi = SO3Convolution(nfeature_in=self.nfeature_in, nfeature_out=self.nfeature_hidden, b_in=self.b_in, b_out=self.b_hidden, grid=self.grid_so3)
        self.Whi = SO3Convolution(nfeature_in=self.nfeature_hidden, nfeature_out=self.nfeature_hidden, b_in=self.b_hidden, b_out=self.b_hidden, grid=self.grid_so3)
        self.Wxf = SO3Convolution(nfeature_in=self.nfeature_in, nfeature_out=self.nfeature_hidden, b_in=self.b_in, b_out=self.b_hidden, grid=self.grid_so3)
        self.Whf = SO3Convolution(nfeature_in=self.nfeature_hidden, nfeature_out=self.nfeature_hidden, b_in=self.b_hidden, b_out=self.b_hidden, grid=self.grid_so3)
        self.Wxc = SO3Convolution(nfeature_in=self.nfeature_in, nfeature_out=self.nfeature_hidden, b_in=self.b_in, b_out=self.b_hidden, grid=self.grid_so3)
        self.Whc = SO3Convolution(nfeature_in=self.nfeature_hidden, nfeature_out=self.nfeature_hidden, b_in=self.b_hidden, b_out=self.b_hidden, grid=self.grid_so3)
        self.Wxo = SO3Convolution(nfeature_in=self.nfeature_in, nfeature_out=self.nfeature_hidden, b_in=self.b_in, b_out=self.b_hidden, grid=self.grid_so3)
        self.Who = SO3Convolution(nfeature_in=self.nfeature_hidden, nfeature_out=self.nfeature_hidden, b_in=self.b_hidden, b_out=self.b_hidden, grid=self.grid_so3)

        self.Wci = None
    
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        """
        :param: x; [batch, nfeature_hidden, beta, alpha, gamma]
        :param: h,c [batch, nfeature_hidden, beta, alpha, gamma]
      
        """
        # print("cell", x.shape, h.shape, self.Wci.shape)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)  # i_t, ingate
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)  # ｆ_t forgetgate
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))  # cell state
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)  # output gate
        ch = co * torch.tanh(cc)  # hidden state also known as output vector
    
        return ch, cc

    def init_hidden(self, batch, hidden, shape, device="cuda:0"):
        if self.Wci is None:
            # print("Initial once for Wci", shape, self.Wci)
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).to(device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).to(device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).to(device)
            # format:(batch_size, channels, height, width)
        else:
            assert shape[0] == self.Wci.size(2), 'Input size Mismatched!'


class SO3ConvLstm(nn.Module):
    def __init__(self, nfeature_in, nfeature_hidden, b_in, b_hidden, Sequences):
        super().__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_hidden = nfeature_hidden
        self.Sequences = Sequences
        self.b_in = b_in
        self.b_hidden = b_hidden
        self._all_layers = []
        self.cell = SO3ConvLSTMCell(self.nfeature_in, self.nfeature_hidden, self.b_in, self.b_hidden)

    def forward(self, inputs, states):
        '''
        :param inputs: [batch, S, nfeature_in, beta, alpha, gamma]
        :param hidden_state: (hx: [batch, S, nfeature_in, beta, alpha, gamma], cx: [batch, S, nfeature_in, beta, alpha, gamma])
        :return:

        '''
        if states is None:
            B, _, C_in, beta, alpha, gamma = inputs.shape
            h = torch.zeros(B, self.nfeature_hidden, beta, alpha, gamma).to(inputs.device)
            c = torch.zeros(B, self.nfeature_hidden, beta, alpha, gamma).to(inputs.device)
        else:
            h, c = states
        # self.cell = self.cell.to(inputs.device)
        outputs = []

        for i in range(self.Sequences):
    
            if inputs is None:
                B, C_out, beta, alpha, gamma = h.shape
                x = torch.zeros((h.size(0), self.nfeature_in, beta, alpha, gamma)).to(h.device)
            else:
                x = inputs[:, i]
            if i == 0:
                B, C_in, beta, alpha, gamma = x.shape
                self.cell.init_hidden(batch=B, hidden=self.nfeature_hidden, shape=(beta, alpha, gamma), device=x.device)
            h, c = self.cell(x, h, c)
            outputs.append(h) 
        return torch.stack(outputs).permute(1, 0, 2, 3, 4, 5).contiguous(), (h, c)  # (S, B, C, beta, alpha, gamma) -> (B, S, C, beta, alpha, gamma)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        self.Sequences = config.sequences_in
        self.stages = len(config.encoder[0])
        for idx, (params, lstm) in enumerate(zip(config.encoder[0], config.encoder[1]), 1):
            setattr(self, "stage"+'_'+str(idx), self._make_layer(params))
            setattr(self, 'lstm'+'_'+str(idx), lstm)

    def _make_layer(self, params):
        layers = []
        for layer_name, v in params.items():
            if 'so3' in layer_name:
        
                # layers.append(nn.BatchNorm3d(v[0], affine=True))
                grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * v[3], n_beta=1, n_gamma=1)
                layers.append(SO3Convolution(nfeature_in=v[0], nfeature_out=v[1],
                                             b_in=v[2], b_out=v[3], grid=grid))

                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 's2' in layer_name:
                # layers.append(nn.Dropout3d(p=0.5))
            
                # layers.append(nn.BatchNorm3d(v[0], affine=True))
                # print(v)
                grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * v[3], n_beta=1)
                layers.append(S2Convolution(nfeature_in=v[0], nfeature_out=v[1], b_in=v[2], b_out=v[3], grid=grid))
                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 'deconv' in layer_name:
                layers.append(nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3], padding=v[4]))
                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 'conv' in layer_name:
                layers.append(nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                        kernel_size=v[2], stride=v[3], padding=v[4]))
                # layers.append(nn.Dropout2d(p=0.5))
                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                raise NotImplementedError
        return nn.Sequential(*layers)

    def forward_stage(self, input, net, lstm, idx=0):
        if len(input.shape) == 6:
            B, S, C, beta, alpha, gamma = input.shape
            input = input.view(B * S, C, beta, alpha, gamma)
        else:
            B, S, C, beta, alpha = input.shape
            input = input.view(B * S, C, beta, alpha)

        input = net(input)
        input = input.view(B, S, *input.shape[1:])  
        out, (h, c) = lstm(input, None)

        return out, (h, c)

    def forward(self, input):
        '''
        :param x: [batch, S, nfeature_in, beta, alpha, gamma]
        :param inter_state: [batch, S, nfeature_in, beta, alpha, gammma]
        :return: outputs: (layer_conv_lstm+1, batch, S, nfeature_in, beta, alpha, gamma) in order of layer_conv_lstm
        '''
        assert self.Sequences == input.size()[1], 'Input Sequences Mismatched!'
        hidden_states = []
        for idx in range(1, self.stages + 1):
            # print("Encoder, stage: ", idx)
            # print("encoder: " + str(idx))
            input, state_stage = self.forward_stage(input, getattr(self, 'stage'+'_'+str(idx)), getattr(self, 'lstm'+'_'+str(idx)), idx)
            # print(state_stage[0].shape)
            hidden_states.append(state_stage)
            # print(input.shape)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Sequences = config.sequences_out
        self.stages = len(config.decoder[0])
        self.config = config
        for idx, (params, lstm) in enumerate(zip(config.decoder[0], config.decoder[1])):
            setattr(self, 'lstm' + '_' + str(self.stages - idx), lstm)
            setattr(self, "stage" + '_' + str(self.stages - idx), self._make_layer(params, self.stages - idx))

    def _make_layer(self, params, idx):
        layers = []
        for layer_name, v in params.items():
            if 'so3' in layer_name:
                # layers.append(nn.Dropout3d(p=0.5))
               
                # layers.append(nn.BatchNorm3d(v[0], affine=True))
                grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * v[3], n_beta=1, n_gamma=1)
                layers.append(SO3Convolution(nfeature_in=v[0], nfeature_out=v[1],
                                             b_in=v[2], b_out=v[3], grid=grid))
                # layers.append(nn.BatchNorm3d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 's2' in layer_name:
    
                grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * v[3], n_beta=1)
                layers.append(S2Convolution(nfeature_in=v[0], nfeature_out=v[1], b_in=v[2], b_out=v[3], grid=grid))
                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 'deconv' in layer_name:
                if idx == self.config.idx:
                    layers.append(EndPool(self.config))
                layers.append(nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3], padding=v[4]))
                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 'conv' in layer_name:
               
                layers.append(nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                        kernel_size=v[2], stride=v[3], padding=v[4]))

               
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                if 'sigmoid' in layer_name:
                    layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError
        return nn.Sequential(*layers)

    def forward_stage(self, input, stage, net, lstm, idx=3):
        input, stage = lstm(input, stage)
  

        if len(input.shape) == 6:
            B, S, C, beta, alpha, gamma = input.shape
            input = input.view(B*S, C, beta, alpha, gamma)
        else:
            B, S, C, beta, alpha = input.shape
            input = input.view(B * S, C, beta, alpha)

        input = net(input)
        input = input.view(B, S, *input.shape[1:])
        # print(input.shape)
        return input

    def forward(self, hidden_states):
        '''
        :param hidden_states: (layer_conv_lstm,（h, c）) in order of layer_conv_lstm
        :return:input: (B, S_out, C, beta, alpha, gamma)
        '''
        
        first_str = ["stage_" + str(self.stages), "lstm_" + str(self.stages)]
        input = self.forward_stage(None, hidden_states[-1], getattr(self, first_str[0]),
                                   getattr(self, first_str[1]), idx=4)
        for idx in list(range(1, self.stages))[::-1]:
           
            input = self.forward_stage(input, hidden_states[idx-1], getattr(self, 'stage' + '_' + str(idx)),
                                       getattr(self, 'lstm' + '_' + str(idx)), idx)
           
        return input


class EndPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final = nn.MaxPool3d(kernel_size=(1, 1, config.end_pra), stride=(1, 1, config.end_pra), padding=(0, 0, 0))  # 方案1

    def forward(self, input):
        
        input = self.final(input)
        input = input.squeeze(-1)
       
        return input


class SO3EncoderForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)

        x = x.squeeze(-1)

        return x
