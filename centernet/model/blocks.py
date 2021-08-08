import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolutionBlock(nn.Module):
    
    def __init__(self, input_channels: int, output_channels: int):
        super(DoubleConvolutionBlock, self).__init__()
        self.double_conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_block(x)


class UpSampleBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, bilinear=True):
        super(UpSampleBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                input_channels // 2, input_channels // 2, 2, stride=2)
        self.conv = DoubleConvolutionBlock(
            input_channels, output_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ))
        x = torch.cat([x2, x1], dim=1) if x2 is not None else x1
        return self.conv(x)
