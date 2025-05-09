# -*- coding: utf-8 -*-
from net.fusion_net import PositionalEncoding
from torch import nn
import torch


class VideoMLP(nn.Module):
    def __init__(self, t_len):
        super(VideoMLP, self).__init__()
        self.mlp = nn.Sequential(  # [tlen, 136]
            nn.BatchNorm1d(t_len),
            nn.Linear(136, 1024),
            nn.BatchNorm1d(t_len),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(t_len),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(t_len),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


class VideoLSTM(nn.Module):
    def __init__(self,):
        super(VideoLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=136,
            hidden_size=512,
            num_layers=6,
            batch_first=True
        )

    def forward(self, x):
        x, (h_n, h_c) = self.rnn(x, None)
        return x


class VideoTransformer(nn.Module):
    def __init__(self, t_len):
        super(VideoTransformer, self).__init__()
        self.bn = nn.BatchNorm1d(t_len)
        self.position_encoder = PositionalEncoding(136)
        encoder_layer = nn.TransformerEncoderLayer(d_model=136, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(136, 512)
        self.bn_out = nn.BatchNorm1d(t_len)

    def forward(self, x):
        x = self.bn(x)
        x = self.position_encoder(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        # x = self.bn_out(x)
        return x
