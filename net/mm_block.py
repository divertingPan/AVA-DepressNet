import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
PyTorch实现Transformer模型细节: https://zhuanlan.zhihu.com/p/127030939
pytorch中的transformer: https://zhuanlan.zhihu.com/p/107586681
Transformer Layers: https://pytorch.org/docs/stable/nn.html#transformer-layers
torch.nn.Transformer解读与应用: https://blog.csdn.net/qq_43645301/article/details/109279616
transformer中的positional encoding(位置编码): https://blog.csdn.net/Flying_sfeng/article/details/100996524
"""


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


class AAEBlock(nn.Module):
    """    Audio Attention Encoder
    """

    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=2):
        super(AAEBlock, self).__init__()
        self.k = k

        self.group_conv = nn.Conv2d(in_channels, inner_channels, kernel_size,
                                    stride, padding, dilation, groups=k)
        self.bn = nn.BatchNorm2d(inner_channels)
        self.relu = nn.ReLU(inplace=True)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.position_encoder = PositionalEncoding(inner_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=inner_channels, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.softmax = SoftMax(2)

    def forward(self, fft, mfcc):
        fft = self.group_conv(fft)  # [batch, channel, time, point]
        fft = self.bn(fft)
        fft = self.relu(fft)
        mfcc = self.group_conv(mfcc)
        mfcc = self.bn(mfcc)
        mfcc = self.relu(mfcc)

        att_fft = self.adaptive_pool(fft)  # [batch, channel, time, 1]
        att_mfcc = self.adaptive_pool(mfcc)

        att_fft = torch.squeeze(att_fft, -1)  # [batch, channel, time]
        att_mfcc = torch.squeeze(att_mfcc, -1)
        att_fft = att_fft.transpose(1, 2)  # [batch, time, channel]
        att_mfcc = att_mfcc.transpose(1, 2)

        att = self.position_encoder(att_fft + att_mfcc)
        att = self.transformer_encoder(att)
        att = self.softmax(att)  # [batch, time, channel]

        att = att.transpose(1, 2)  # [batch, channel, time]

        return fft, mfcc, att


class VisAEBlock(nn.Module):
    """    Visual Attention Encoder
    """
    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=2):
        super(VisAEBlock, self).__init__()
        self.k = k

        self.group_conv = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=k)
        self.bn = nn.BatchNorm3d(inner_channels)
        self.relu = nn.ReLU(inplace=True)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.position_encoder = PositionalEncoding(inner_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=inner_channels, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.softmax = SoftMax(2)

    def forward(self, x):
        x = self.group_conv(x)  # [batch, channel, time, h, w]
        x = self.bn(x)
        x = self.relu(x)

        att = self.adaptive_pool(x)  # [batch, channel, time, 1, 1]
        att = torch.squeeze(att, -1)
        att = torch.squeeze(att, -1)  # [batch, channel, time]

        att = att.transpose(1, 2)  # [batch, time, channel]
        att = self.position_encoder(att)
        att = self.transformer_encoder(att)
        att = self.softmax(att)  # [batch, time, channel]

        att = att.transpose(1, 2)  # [batch, channel, time]

        return x, att


class AVACBlock(nn.Module):
    """    Audio-Visual Attention Combiner
    """
    def __init__(self, audio_in_channels, visual_in_channels, inner_channels,
                 kernel_size, audio_stride, visual_stride, padding, dilation, k=2):
        super(AVACBlock, self).__init__()
        self.AAE_layer = AAEBlock(audio_in_channels, inner_channels,
                                  kernel_size, audio_stride, padding, dilation, k)
        self.VisAE_layer = VisAEBlock(visual_in_channels, inner_channels,
                                      kernel_size, visual_stride, padding, dilation, k)
        self.softmax = SoftMax(1)

    def forward(self, fft, mfcc, img):
        fft, mfcc, att_audio = self.AAE_layer(fft, mfcc)
        img, att_visual = self.VisAE_layer(img)

        att = self.softmax(att_audio + att_visual)  # [batch, channel, time]
        att = torch.unsqueeze(att, -1)  # [batch, channel, time, 1]
        fft = fft * att
        mfcc = mfcc * att
        att = torch.unsqueeze(att, -1)  # [batch, channel, time, 1, 1]
        img = img * att

        return fft, mfcc, img, att_audio, att_visual


if __name__ == '__main__':

    # net = AAEBlock(in_channels=1, inner_channels=16, kernel_size=3,
    #                stride=1, padding=1, dilation=1, k=1).to('cuda')
    fft_train = torch.rand(5, 1, 64, 1025).to('cuda')    # [batch, channel, time, point]
    mfcc_train = torch.rand(5, 1, 64, 32).to('cuda')

    # fft_feature, mfcc_feature, att_vector = net(fft_train, mfcc_train)

    # print('fft_feature {}, mfcc_feature {}, att_vector {}'.format(fft_feature.shape, mfcc_feature.shape, att_vector.shape))

    # encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
    # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
    # position_encoding = PositionalEncoding(1024)
    # src = torch.rand(64, 5, 1024)
    # src = position_encoding(src)
    # out = transformer_encoder(src)
    # print(src.shape)
    # print(out.shape)

    # net = VisAEBlock(in_channels=3, inner_channels=16, kernel_size=3,
    #                  stride=1, padding=1, dilation=1, k=1).to('cuda')
    img_train = torch.rand(5, 3, 64, 224, 224).to('cuda')

    # image_feature, image_att = net(img_train)
    # print('image_feature {}, image_att {}'.format(image_feature.shape, image_att.shape))

    net = AVACBlock(audio_in_channels=1, visual_in_channels=3,
                    inner_channels=16, kernel_size=3,
                    audio_stride=(1, 2), visual_stride=(1, 2, 2), padding=1, dilation=1, k=1).to('cuda')
    fft_feature, mfcc_feature, image_feature, att_audio, att_visual = net(fft_train, mfcc_train, img_train)
    print('fft_feature {}, mfcc_feature {}, '
          'image_feature {}, att_audio {},'
          ' att_visual {}'.format(fft_feature.shape, mfcc_feature.shape,
                                  image_feature.shape, att_audio.shape, att_visual.shape))
    """
    fft_feature torch.Size([5, 16, 64, 1025]),
    mfcc_feature torch.Size([5, 16, 64, 32]),
    image_feature torch.Size([5, 16, 64, 224, 224]),
    att_audio torch.Size([5, 16, 64]),
    att_visual torch.Size([5, 16, 64])
    """
