# -*- encoding: utf-8 -*-
"""

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

