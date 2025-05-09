import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.mult_model import MULTModel


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SoftMax(nn.Module):
    def __init__(self, dim):
        super(SoftMax, self).__init__()
        self.dim = dim

    def forward(self, x):
        if len(x.shape) > 1:
            x = F.softmax(x, dim=self.dim)
        else:
            x = torch.sigmoid(x)
        return x


class MultimodalTransformer(nn.Module):
    def __init__(self, data_len, time_len, mode, dataset):
        super(MultimodalTransformer, self).__init__()
        self.mode = mode
        self.dataset = dataset
        self.data_len = data_len
        self.time_len = time_len
        self.position_encoder = PositionalEncoding(data_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=data_len, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.softmax = SoftMax(2)
        self.bn0 = nn.BatchNorm1d(time_len)
        self.regression = nn.Linear(data_len, 1)
        self.regression_1 = nn.Linear(data_len, 2048)
        self.bn1 = nn.BatchNorm1d(time_len)
        self.regression_2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(time_len)
        self.regression_3 = nn.Linear(1024, 1)
        self.bn3 = nn.BatchNorm1d(time_len)
        self.softmax_regression = SoftMax(0)
        self.linear = nn.Linear(time_len, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.relu(self.bn0(x))
        # encoding the slice with internal information
        # x: [BATCHSIZE, T_LEN, audio_feature_len]
        x = self.position_encoder(x)
        x = self.transformer_encoder(x)
        # x = self.softmax(x)  # [BATCHSIZE, T_LEN, audio_feature_len]
        # x = self.relu(self.bn0(x))

        # regression the single frame result in each time slice
        # x = x.view(-1, self.data_len)  # [T_LEN, audio_feature_len]

        x = self.regression_1(x)  # [BATCHSIZE, T_LEN, 2048]
        # x = self.relu(self.bn1(x))
        x = self.regression_2(x)  # [BATCHSIZE, T_LEN, 1024]
        # x = self.relu(self.bn2(x))
        x = self.regression_3(x)  # [BATCHSIZE, T_LEN, 1]
        # x = self.relu(self.bn3(x))
        # x = self.regression(x)  # [T_LEN, 1]
        # x = self.softmax_regression(x)  # [T_LEN, 1]

        # extract the total score from all information from frames in the slice
        # x = x.view(1, -1)  # [1, T_LEN]
        x = x.view(-1, 1, self.time_len)
        x = self.linear(x)  # [BATCHSIZE, 1, 1]
        x = x.squeeze()
        x = torch.sigmoid(x)

        if self.dataset == 'avec17':
            x = x * 23
        else:
            x = x * 63

        return x


class UnalignedTransformer(nn.Module):
    def __init__(self, feature_size_a, feature_size_v, dataset):
        super(UnalignedTransformer, self).__init__()
        self.dataset = dataset
        self.net = MULTModel(feature_size_a, feature_size_v)

    def forward(self, x_a, x_v):
        x, _ = self.net(x_a, x_v)
        x = torch.sigmoid(x)
        # x = torch.relu(x)
        x = x.squeeze()
        if self.dataset == 'avec17':
            x = x * 23
        else:
            x = x * 63

        return x


class ConcatLayer(nn.Module):
    def __init__(self, feature_size, dataset):
        super(ConcatLayer, self).__init__()
        self.dataset = dataset
        self.net = nn.Linear(feature_size, 1)

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        if self.dataset == 'avec17':
            x = x * 23
        else:
            x = x * 63

        return x
