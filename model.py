# -*- encoding: utf-8 -*-
"""
@ Author  ：
@ File    : model.py
@ Time    ：2021/9/21 11:41
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0 

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        # self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)  

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)  # i_t, ingate
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)  # f_t forgetgate
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))  # cell state
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)  # output gate
        ch = co * torch.tanh(cc)  # hidden state also known as output vector
        return ch, cc

    def init_hidden(self, batch, hidden, shape):
        if self.Wci is None:
            # print("Initial once for Wci", shape, self.Wci)
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            # format:(batch_size, channels, height, width)
        else:
            # print(self.Wci.size(), shape)
            assert shape[0] == self.Wci.size()[2], 'Input size Mismatched!'


class ConvLstm(nn.Module):
    def __init__(self, input_channels, hidden_channels, Sequences, kernel_size):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels 
        self.Sequences = Sequences
        self.kernel_size = kernel_size
        self._all_layers = []
        self.cell = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, states):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        # Batch, Sequence, Channels, Height, Width

        '''
        if states is None:
            B, _, C_in, H, W = inputs.shape
            h = torch.zeros(B, self.hidden_channels, H, W).to(inputs.device)
            c = torch.zeros(B, self.hidden_channels, H, W).to(inputs.device)
        else:
            h, c = states
        outputs = []

        for i in range(self.Sequences):
            # S——Sequence
            if inputs is None:
                B, C_out, H, W = h.shape
                x = torch.zeros((h.size(0), self.input_channels, H, W)).to(h.device)
            else:
                x = inputs[:, i]
            if i == 0:
                B, C_in, H, W = x.shape
                self.cell.init_hidden(batch=B, hidden=self.hidden_channels, shape=(H, W))
            h, c = self.cell(x, h, c)
            outputs.append(h)

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous(), (h, c)  # (S, B, C, H, W) -> (B, S, C, H, W)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.layers = []
        self.Sequences = config.sequences_in
        self.stages = len(config.encoder[0])
        for idx, (params, lstm) in enumerate(zip(config.encoder[0], config.encoder[1]), 1):
            setattr(self, "stage"+'_'+str(idx), self._make_layer(params))
            setattr(self, 'lstm'+'_'+str(idx), lstm)
            # self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, params):
        layers = []
        for layer_name, v in params.items():
            if 'conv' in layer_name:
                layers.append(nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                        kernel_size=v[2], stride=v[3], padding=v[4]))
                # layers.append(nn.Dropout2d(p=0.5))
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
            else:
                raise NotImplementedError
        return nn.Sequential(*layers)

    def forward_stage(self, input, net, lstm):
        B, S, C, H, W = input.shape
        input = input.view(B*S, C, H, W)

        input = net(input)
        input = input.view(B, S, input.shape[1], input.shape[2], input.shape[3])  # conv后的还原

        out, (h, c) = lstm(input, None)
        return out, (h, c)

    def forward(self, input):
        '''
        :param input: (B, S, C, H, W)
        :return: hidden_states: (layer_conv_lstm,（h, c）) in order of layer_conv_lstm
        '''
        assert self.Sequences == input.size()[1], 'Input Sequences Mismatched!'
        hidden_states = []
        for idx in range(1, self.stages + 1):
            # print("encoder: " + str(idx))
            input, state_stage = self.forward_stage(input, getattr(self, 'stage'+'_'+str(idx)),
                                                    getattr(self, 'lstm'+'_'+str(idx)))
            # print(state_stage[0].shape)
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Decoder(nn.Module):
    def __init__(self, config):
        self.Sequences = config.sequences_out
        super().__init__()
        self.stages = len(config.decoder[0])
        for idx, (params, lstm) in enumerate(zip(config.decoder[0], config.decoder[1])):
            setattr(self, 'lstm' + '_' + str(self.stages - idx), lstm)
            setattr(self, "stage" + '_' + str(self.stages - idx), self._make_layer(params))

    def _make_layer(self, params):
        layers = []
        for layer_name, v in params.items():
            if 'deconv' in layer_name:
                layers.append(nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3], padding=v[4]))
                # layers.append(nn.Dropout2d(p=0.5))
                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif 'conv' in layer_name:
                # layers.append(nn.Dropout2d(p=0.5))
                layers.append(nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                        kernel_size=v[2], stride=v[3], padding=v[4]))

                # layers.append(nn.BatchNorm2d(v[1]))
                if 'relu' in layer_name:
                    layers.append(nn.ReLU(inplace=True))
                if 'leaky' in layer_name:
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                raise NotImplementedError
        return nn.Sequential(*layers)

    def forward_stage(self, input, stage, net, lstm):
        input, stage = lstm(input, stage)
        B, S, C, H, W = input.shape
        input = input.view(B*S, C, H, W)

        input = net(input)
        input = input.view(B, S, input.shape[1], input.shape[2], input.shape[3])  # conv后的还原
        # print(input.shape)
        return input

    def forward(self, hidden_states):
        '''
        :param hidden_states: (layer_conv_lstm,（h, c）) in order of layer_conv_lstm
        :return: input: (B, S, C, H, W)
        '''
        # assert self.Sequences==encoder_outputs.size()[2], 'Input sequence Mismatched!'
        # print("decoder: " + str(3))
        input = self.forward_stage(None, hidden_states[-1], getattr(self, 'stage_' + str(self.stages)),
                                      getattr(self, 'lstm_' + str(self.stages)))
        for idx in list(range(1, self.stages))[::-1]:
            # print("decoder: " + str(idx))
            input = self.forward_stage(input, hidden_states[idx-1], getattr(self, 'stage' + '_' + str(idx)),
                                          getattr(self, 'lstm' + '_' + str(idx)))
            # print(input.shape)
        return input


class EncoderForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        # self.final = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 方案1
        # self.final = nn.Dropout2d(p=0.5)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        # x = self.final(x)
        return x


if __name__ == '__main__':
    # from thop import profile
    from config import config
    model = EncoderForecaster(config)
    model = model.cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.ones(config.batch, config.sequences_in, 1, config.height, config.width)
    print(input.shape)
    input = input.to(device)
    out = model(input)
    print(out.size())
