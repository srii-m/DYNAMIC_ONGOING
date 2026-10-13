import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3, 3), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UnifiedISLNet(nn.Module):
    """
    Unified 2D CNN for both static (T=1) and dynamic (T=60) ISL signs.
    Input: (B, 1, T, F)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=12, max_time=60, features=252):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4 = ConvBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x);  x = self.pool1(x)
        x = self.conv2(x);  x = self.pool2(x)
        x = self.conv3(x);  x = self.pool3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x