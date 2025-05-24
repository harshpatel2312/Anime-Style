import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Reduces edge artifacts
            nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels,  kernel_size = 3, stride = 1, padding = 0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) # Add new features learnt
