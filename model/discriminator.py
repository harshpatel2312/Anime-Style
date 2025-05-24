import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_nc = 3, ndf = 64):
        super(Discriminator, self).__init__()
        
        # input_nc: number of input channels (e.g., 3 for RGB)
        # ndf: number of filters in the first conv layer

        layers = [
            # Layer 1
            nn.Conv2d(input_nc, ndf, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        layers += [
            # Layer 2
            nn.Conv2d(ndf, ndf*2, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        layers += [
            # Layer 3
            nn.Conv2d(ndf*2, ndf*4, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        layers += [
            # Layer 4
            nn.Conv2d(ndf*4, ndf*8, kernel_size = 4, stride = 1, padding = 1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace = True)
        ]

        layers += [
            # Final layer: 1 channel prediction map (PatchGAN)
            nn.Conv2d(ndf*8, 1, kernel_size = 4, stride = 1, padding = 1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        