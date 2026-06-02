# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class BottleNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_factor,
                 stride=1):
        super().__init__()

        mid_channels = in_channels * expansion_factor

        layers = []
        
        # in data: 24, 56, 56
        if expansion_factor != 1: # two relu in a row lleads to zero output
            layers.extend([
                # expansion
                nn.Conv2d(in_channels,
                        mid_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0,
                        bias=False),
                nn.BatchNorm2d(mid_channels), # 144, 56, 56
                nn.ReLU6(inplace=True),
            ])
            
        layers.extend([
            # deepwise conv
            nn.Conv2d(mid_channels, 
                      mid_channels, 
                      kernel_size=3, 
                      stride=stride,
                      padding=1,
                      groups=mid_channels,
                      bias=False),
            nn.BatchNorm2d(mid_channels), # 144 56, 56
            nn.ReLU6(inplace=True),
            # projection
            nn.Conv2d(mid_channels, 
                      out_channels, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            # NO RELU HERE, "linear bottleneck"
        ])

        self.main_path = nn.Sequential(*layers)

        # if channel and shape doesn't match, do not use resudual
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.main_path(x) 
        return self.main_path(x)
        

# %%
class MobileNetV2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=1000,
                 ):
        super().__init__()

        params = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1], 
        ]

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True), # bs, 32, 112
        ]

        in_channels = 32
        for exp_factor, out_channels, repeat, stride in params:
            for i in range(repeat):
                stride = stride if i == 0 else 1
                layers.append(
                    BottleNeck(in_channels=in_channels,
                               out_channels=out_channels,
                               expansion_factor=exp_factor,
                               stride=stride, # bs, out_chanels, h/2, w/2
                ))
                in_channels = out_channels
        
        layers.extend([
            nn.Conv2d(in_channels, 
                      out_channels=1280, 
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(1280), # bs, 1280, 7, 7
            nn.ReLU6(inplace=True),
        ])

        layers.extend([
            nn.AdaptiveAvgPool2d(1), # bs, 1280, 1, 1
            nn.Conv2d(in_channels=1280, 
                      out_channels=num_classes, 
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), # n_classes, 1, 1
            # servers as FC so no BN, allow bias
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return torch.flatten(x, 1)

