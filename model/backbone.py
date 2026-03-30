import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # Expansion phase
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU())

        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU())

        # Squeeze and excitation phase
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class Backbone(nn.Module):
    def __init__(self, in_channels=3, expand_ratio=6):
        super().__init__()
        self.init_weights()
        self.expand_ratio = expand_ratio
        # 按照MobileNetV2的配比，通道数逐渐增加

        # 2x下采样
        self.backbone_block_2x = nn.Sequential(
            conv3x3(in_channels, 32, 2),
            InvertedResidual(32, 16, stride=1, expand_ratio=self.expand_ratio)
        )

        # 4x下采样
        self.backbone_block_4x = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=self.expand_ratio),
            InvertedResidual(24, 24, stride=1, expand_ratio=self.expand_ratio)
        )

        # 8x下采样
        self.backbone_block_8x = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=self.expand_ratio),
            InvertedResidual(32, 32, stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(32, 32, stride=1, expand_ratio=self.expand_ratio)
        )

        # 16x下采样
        self.backbone_block_16x = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=self.expand_ratio),
            InvertedResidual(64, 64, stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(64, 64, stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(64, 96, stride=1, expand_ratio=self.expand_ratio)
        )

        # 上采样路径：16x -> 8x
        self.up_16x_to_8x = nn.Sequential(
            nn.ConvTranspose2d(96, 96, 4, padding=1, stride=2, bias=False, groups=96),
            conv1x1(96, 96)
        )

        # 上采样路径：8x -> 4x
        self.up_8x_to_4x = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2, bias=False, groups=128),
            conv1x1(128, 128)
        )

        # 融合层：将不同尺度的特征融合
        self.fusion_8x = conv1x1(96 + 32, 128)  # 融合8x上采样和4x下采样
        self.fusion_4x = conv1x1(128 + 24, 128)  # 融合8x上采样和4x下采样

        self.channels = 128  # 4x

    def forward(self, x):
        # 下采样路径
        x_2x = self.backbone_block_2x(x)      # 1/2
        x_4x = self.backbone_block_4x(x_2x)   # 1/4
        x_8x = self.backbone_block_8x(x_4x)   # 1/8
        x_16x = self.backbone_block_16x(x_8x) # 1/16

        # 上采样路径，逐步融合
        # 16x -> 8x
        x_8x_up = self.up_16x_to_8x(x_16x)
        x_8x_fused = self.fusion_8x(torch.cat([x_8x_up, x_8x], 1))

        # 8x -> 4x
        x_4x_up = self.up_8x_to_4x(x_8x_fused)
        x_4x_fused = self.fusion_4x(torch.cat([x_4x_up, x_4x], 1))

        return x_4x_fused
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

class BevNeck(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=32, expand_ratio=3):
        super().__init__()
        self.conv_1x = nn.Sequential(
            conv3x3(in_channels, inner_channels, stride=1),
            InvertedResidual(inner_channels, inner_channels, stride=1, expand_ratio=expand_ratio)
        )
        self.conv_2x = nn.Sequential(
            InvertedResidual(inner_channels, inner_channels * 2, stride=2, expand_ratio=expand_ratio),
            InvertedResidual(inner_channels * 2, inner_channels * 2, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(inner_channels * 2, inner_channels * 2, stride=1, expand_ratio=expand_ratio),
        )
        self.conv_4x = nn.Sequential(
            InvertedResidual(inner_channels * 2, inner_channels * 3, stride=2, expand_ratio=expand_ratio),
            InvertedResidual(inner_channels * 3, inner_channels * 3, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(inner_channels * 3, inner_channels * 3, stride=1, expand_ratio=expand_ratio),
        )
        self.conv_8x = nn.Sequential(
            InvertedResidual(inner_channels * 3, inner_channels * 3, stride=2, expand_ratio=expand_ratio),
            InvertedResidual(inner_channels * 3, inner_channels * 3, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(inner_channels * 3, inner_channels * 3, stride=1, expand_ratio=expand_ratio),
        )

        self.up_8x_to_4x = nn.Sequential(
            nn.ConvTranspose2d(inner_channels * 3, inner_channels * 3, 4, padding=1, stride=2, bias=False, groups=inner_channels * 3),
            conv1x1(inner_channels * 3, out_channels)
        )
        self.up_4x_to_2x = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 4, padding=1, stride=2, bias=False, groups=out_channels),
            conv1x1(out_channels, out_channels)
        )
        self.up_2x_to_1x = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 4, padding=1, stride=2, bias=False, groups=out_channels),
            conv1x1(out_channels, out_channels)
        )

        self.fusion_8x_4x = conv1x1(inner_channels * 3 + out_channels, out_channels)
        self.fusion_4x_2x = conv1x1(out_channels + inner_channels * 2, out_channels)
        self.fusion_2x_1x = conv1x1(out_channels + inner_channels, out_channels)

    def forward(self, x):
        x_1x = self.conv_1x(x)
        x_2x = self.conv_2x(x_1x)
        x_4x = self.conv_4x(x_2x)
        x_8x = self.conv_8x(x_4x)

        x_4x_up = self.up_8x_to_4x(x_8x)
        x_4x_fused = self.fusion_8x_4x(torch.cat([x_4x_up, x_4x], 1))

        x_2x_up = self.up_4x_to_2x(x_4x_fused)
        x_2x_fused = self.fusion_4x_2x(torch.cat([x_2x_up, x_2x], 1))

        x_1x_up = self.up_2x_to_1x(x_2x_fused)
        x_1x_fused = self.fusion_2x_1x(torch.cat([x_1x_up, x_1x], 1))
        return x_1x_fused
        