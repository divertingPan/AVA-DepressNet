import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, x):
        feature = x.size(2)
        if feature > 1:
            x = F.softmax(x, dim=2)
        else:
            x = torch.sigmoid(x)
        return x


class AttentionSpatiotemporalBlock(nn.Module):
    """Spatio-Temporal attention vector fusing attention block

    Arguments:
        in_channels (int): the dim of the whole tensor flowing into this block
        inner_channels (int): the dim of total tensor among k-group in two block
        k (int): inner groups of each block
    """

    def __init__(self, in_channels, inner_channels, kernel_size, stride, padding, dilation, k=1):
        super(AttentionSpatiotemporalBlock, self).__init__()
        self.k = k

        # temporal block before attention vector
        self.group_conv_temporal = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                             stride, padding, dilation, groups=k)
        self.bn_temporal = nn.BatchNorm3d(inner_channels)
        self.adaptive_pool_temporal = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc_temporal = nn.Conv3d(inner_channels, inner_channels, kernel_size=1, groups=k)

        # spatial block before attention vector
        self.group_conv_spatial = nn.Conv3d(in_channels, inner_channels, kernel_size,
                                            stride, padding, dilation, groups=k)
        self.bn_spatial = nn.BatchNorm3d(inner_channels)
        self.adaptive_pool_spatial = nn.AdaptiveAvgPool3d((1, None, None))
        self.fc_spatial = nn.Conv3d(inner_channels, inner_channels, 1, groups=k)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = SoftMax()

    def forward(self, x):
        # temporal block before attention vector
        x_t = self.group_conv_temporal(x)
        # x_t = self.bn_temporal(x_t)
        x_t = self.relu(x_t)
        attention_t = self.adaptive_pool_temporal(x_t)
        attention_t = self.fc_temporal(attention_t).view(x_t.size(0), self.k, x_t.size(1) // self.k, x_t.size(2), 1, 1)
        # print('attention_t size:', attention_t.shape)  [batch, k, feature, T, 1, 1]
        attention_t = self.softmax(attention_t)

        # spatial block before attention vector
        x_s = self.group_conv_spatial(x)
        # x_s = self.bn_spatial(x_s)
        x_s = self.relu(x_s)
        attention_s = self.adaptive_pool_spatial(x_s)
        attention_s = self.fc_spatial(attention_s).view(x_s.size(0), self.k, x_s.size(1) // self.k, 1, x_s.size(3), x_s.size(4))
        # print('attention_s size:', attention_s.shape)  [batch, k, feature, 1, H, W]
        attention_s = self.softmax(attention_s)

        x = attention_t * attention_s
        x = x.view(x.size(0), x.size(2) * self.k, x.size(3), x.size(4), x.size(5))
        x = x * (x_s + x_t)

        return x


class EmotionNet(nn.Module):
    def __init__(self, ):
        super(EmotionNet, self).__init__()
        self.conv = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.sta_block1 = AttentionSpatiotemporalBlock(in_channels=32, inner_channels=64,
                                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.sta_block2 = AttentionSpatiotemporalBlock(in_channels=64, inner_channels=128,
                                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.feature = nn.Conv3d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.transpose(1, 2)  # [1, 3, 1024, 64, 64]
        x = self.conv(x)  # [1, 32, 1024, 32, 32]
        x = self.sta_block1(x)  # [1, 64, 1024, 32, 32]
        x = self.sta_block2(x)  # [1, 128, 1024, 32, 32]
        x = self.pooling(x)  # [1, 128, 1024, 1, 1]
        x = self.feature(x)  # [1, 512, 1024, 1, 1]
        x = x.transpose(1, 2)  # [1, 1024, 512, 1, 1]
        return x


if __name__ == '__main__':
    # from torchkeras import summary
    import os
    # net = EmotionNet()
    # summary(net, (3, 1024, 64, 64))
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x = torch.rand((1024, 3, 64, 64)).cuda()
    net = EmotionNet().cuda()
    x = net(x)
    # print(x.shape)
