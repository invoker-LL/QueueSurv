import torch.nn as nn
import torch

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()

        # 使用卷积层增加通道数并逐渐压缩空间维度
        k_size=3
        pad=1
        self.conv1 = nn.Conv2d(32, 128, kernel_size=k_size, stride=2, padding=pad)

        self.norm1 = nn.LayerNorm([128, 7, 7])  # LayerNorm after the conv layer
        self.act1 = nn.GELU()  # GELU activation

        self.conv2 = nn.Conv2d(128, 256, kernel_size=k_size, stride=2, padding=pad)
        self.norm2 = nn.LayerNorm([256, 4, 4])
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(256, 512, kernel_size=k_size, stride=2, padding=pad)
        self.norm3 = nn.LayerNorm([512, 2, 2])
        self.act3 = nn.GELU()

        self.conv4 = nn.Conv2d(512, 1024, kernel_size=k_size, stride=2, padding=pad)
        self.norm4 = nn.LayerNorm([1024, 1, 1])
        self.act4 = nn.GELU()

        # self.conv5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 第一层卷积+GroupNorm+激活
        with torch.no_grad():
        # if True:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.act1(x)

        # 第二层卷积+GroupNorm+激活
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        # 第三层卷积+GroupNorm+激活
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)

        # 第四层卷积+GroupNorm+激活
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act4(x)

        # 第五层卷积，输出最终结果
        # x = self.conv5(x)

        return x