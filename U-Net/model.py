import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stripT import Stripformer

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,  kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        
        out += residual  # Add the residual to the output
        out = self.relu(out)
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2, stride=1),
            #DoubleConv(in_channels, out_channels, stride=(1,2))
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(2,1)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels,mid_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2,1), stride=(2,1)))
        self.conv = ResidualBlock(mid_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=2)
        return self.conv(x)

#dva kon sloja, bez relu-a na kraju
class MusicSep(nn.Module):
    def __init__(self):
        super(MusicSep, self).__init__()

        self.residual0 = (ResidualBlock(4,4))
        
        #1st block
        self.down1 = (Down(4,4))
        self.residual1 = (ResidualBlock(4,32))
        
        #2nd block
        self.down2 = (Down(32,32))
        self.residual2 = (ResidualBlock(32,48))

        #3rd block
        self.down3 = (Down(48,48))
        self.residual3 = (ResidualBlock(48, 64))

#### TO DO: transformeri
# bottleneck
        self.trans = Stripformer()

        #going up!

        self.residual4 = (ResidualBlock(64, 48))

        #self.up1 = (Up(64,64+64,48))
        
        self.up2 = (Up(48, 48, 32))
        # self.residual5 = (ResidualBlock(48,32))

        self.up3 = (Up(32, 32, 4))
        self.residual6 = (ResidualBlock(4,4))


    def forward(self, x):
        
        out0 = self.residual0(x)

        out1 = self.down1(out0)
        out1 = self.residual1(out1)

        out2 = self.down2(out1)
        out2 = self.residual2(out2)

        out3 = self.down3(out2)
        out3 = self.residual3(out3)
        
#trans
        out_t = self.trans(out3)

        #out4 = self.up1(out_t, out3)
        out4 = self.residual4(out_t)

        out5 = self.up2(out4, out2)
        # out5 = self.residual5(out5)

        out6 = self.up3(out5, out1)
        # out6 = self.residual6(out6) 

        out7 = self.residual6(out6)
    
        return out7



