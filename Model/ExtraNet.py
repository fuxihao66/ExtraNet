import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
import torch
import numpy as np
import torch.nn as nn
from config import mdevice


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None ,kernel_size=3,padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=kernel_size,
                               padding=padding)
    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)
class LWGatedConv2D(nn.Module):
    def __init__(self, input_channel1, output_channel, pad, kernel_size, stride):
        super(LWGatedConv2D, self).__init__()

        self.conv_feature = nn.Conv2d(in_channels=input_channel1, out_channels=output_channel, kernel_size=kernel_size,
                                      stride=stride, padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=input_channel1, out_channels=1, kernel_size=kernel_size, stride=stride,
                      padding=pad),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # inputs = inputs * mask
        newinputs = self.conv_feature(inputs)
        mask = self.conv_mask(inputs)

        return newinputs*mask
class DownLWGated(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = LWGatedConv2D(in_channels, in_channels, kernel_size=3, pad=1, stride=2)
        self.conv1 = LWGatedConv2D(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LWGatedConv2D(out_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x= self.downsample(x)
        x= self.conv1(x)
        x = self.relu1(self.bn1(x))
        x= self.conv2(x)
        x = self.relu2(self.bn2(x))
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
      
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class ExtraNet(nn.Module):
    # a tinyer Unet which only has 3 downsample pass
    def __init__(self, n_channels, n_classes, bilinear=False, skip=True):
        super(ExtraNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip = skip

        self.convHis1 = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.convHis2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.convHis3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.lowlevelGated = LWGatedConv2D(32*3, 32, kernel_size=3, stride=1, pad=1)

        self.conv1 = LWGatedConv2D(n_channels, 24, kernel_size=3, stride=1, pad=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LWGatedConv2D(24, 24, kernel_size=3, stride=1, pad=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU(inplace=True)
        self.down1 = DownLWGated(24, 24)
        self.down2 = DownLWGated(24, 32)
        self.down3 = DownLWGated(32, 32)
        
        self.up1 = Up(96, 32)
        self.up2 = Up(56, 24)
        self.up3 = Up(48, 24)
        self.outc = nn.Conv2d(24, n_classes, kernel_size=1)
    def forward(self, x, feature, mask, hisBuffer):

        hisBuffer = hisBuffer.reshape(-1, 4, hisBuffer.shape[-2], hisBuffer.shape[-1])

        hisDown1 = self.convHis1(hisBuffer)
        hisDown2 = self.convHis2(hisDown1)
        hisDown3 = self.convHis3(hisDown2)
        cathisDown3 = hisDown3.reshape(-1, 3*32, hisDown3.shape[-2], hisDown3.shape[-1])  # 64

        motionFeature = self.lowlevelGated(cathisDown3)


        x1=torch.cat([x,x*mask, feature],dim=1)
        x1= self.conv1(x1)
        x1 = self.relu1(self.bn1(x1))
        x1= self.conv2(x1)
        x1 = self.relu2(self.bn2(x1))

        x2= self.down1(x1)
        x3= self.down2(x2)
        x4= self.down3(x3)

        x4 = torch.cat([x4, motionFeature], dim=1)
        res = self.up1(x4, x3)
        res= self.up2(res, x2)
        res= self.up3(res, x1)
        logits = self.outc(res)
        if self.skip:
            logits = logits + x[:, 0:3, :, :]
        return logits