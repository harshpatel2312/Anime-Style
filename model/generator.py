import torch
import torch.nn as nn
from resnet_block import ResidualBlock

class Generator(nn.Module):
    """
    Args:
        input_nc: Input channels
        output_nc: Output channels
        ngf: Number of generator filters
        n_blocks: Number of residual blocks
    """
    def __init__(self, input_nc = 3, output_nc = 3, ngf = 64, n_blocks = 6):
        super(Generator, self).__init__()

        # Initial 7x7 Convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size = 7, stride = 1, padding = 0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace = True)
        ]

        # Downsampling x2 (e.g 512 -> 256 -> 128)
        in_channels = ngf
        out_channels = in_channels * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace = True)
            ]  
            in_channels = out_channels
            out_channels *= 2

        # Residual Blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(in_channels)] # inchannels = 256

        # Upscaling x2 (e.g 128 -> 256 -> 512)
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor = 2, mode = "nearest"),
                nn. Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace = True)
            ]
            in_channels = out_channels
            out_channels //= 2
            
        # Final output layer -> Back to [-1, 1]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, output_nc, kernel_size = 7, stride = 1, padding = 0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
        