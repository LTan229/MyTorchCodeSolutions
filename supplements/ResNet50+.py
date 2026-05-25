
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self,
                 ResBlock,
                 num_blocks,
                 num_classes,
                 in_channels=3,
                 ):
        super().__init__()
        self.block_in_c = 64
        
        self.layers = [ # b, 3, 224, 224
            nn.Conv2d(in_channels, 
                      64, 
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      bias=False), # b, 64, 112, 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # b, 64, 56, 56
        ]

        # stack block to refine features and increase linearity
        # more channels to store higher level features
        self._add_res_block(ResBlock, num_blocks[0], bottleneck_c=64) # b, 256, 56, 56
        self._add_res_block(ResBlock, num_blocks[1], bottleneck_c=128, stride=2) # b, 512, 28, 28
        self._add_res_block(ResBlock, num_blocks[2], bottleneck_c=256, stride=2) # b, 1024, 14, 14
        self._add_res_block(ResBlock, num_blocks[3], bottleneck_c=512, stride=2) # b, 2048, 7, 7

        self.layers.append(nn.AdaptiveAvgPool2d((1, 1))) # b, 2048, 1, 1
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(512 * ResBlock.expansion, num_classes)) # b, num_classes
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
    
    def _add_res_block(self, 
                       ResBlock,
                       num_blocks, 
                       bottleneck_c,
                       stride=1):
        i_downsample = None

        # in channel should equal to out channel
        # b, in_c, h, w -> b, bottleneck, h / stride, w / stride -> b, o_c (larger), h / stride, w/stride
        block_out_c = bottleneck_c * ResBlock.expansion
        if stride != 1 or self.block_in_c != block_out_c:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.block_in_c, 
                          block_out_c, 
                          kernel_size=1, 
                          stride=stride),
                nn.BatchNorm2d(block_out_c)
            )

        self.layers.append(ResBlock(self.block_in_c, 
                                    bottleneck_c, 
                                    i_downsample=i_downsample, 
                                    stride=stride))
        self.block_in_c = block_out_c

        # b, o_c, h, w -> b, bottleneck, h, w -> b, o_c, h, w
        for _ in range(num_blocks - 1):
            self.layers.append(ResBlock(block_out_c, bottleneck_c)) 


class BottleNeck(nn.Module):

    expansion = 4 # expands the output size of the block to match the input of the next block

    def __init__(self,
                 in_channels,
                 neck_channels,
                 i_downsample=None,
                 stride=1):
        super().__init__()

        self.i_downsample = i_downsample

        self.main_path = nn.Sequential(# b, in_c, h, w
            nn.Conv2d(in_channels, # channel compress to reduce computational cost
                      neck_channels, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0,
                      bias=False), # b, o_c, h, w
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(),
            nn.Conv2d(neck_channels,  # feature extraction 
                      neck_channels, 
                      kernel_size=3, 
                      stride=stride,
                      padding=1,
                      bias=False), # b, o_c, h / stride, w / stride
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(),
            nn.Conv2d(neck_channels, # recover channel to enable residual connection
                      neck_channels * self.expansion, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0,
                      bias=False), # b, o_c * self.expansion, h / stride, w / stride
            nn.BatchNorm2d(neck_channels * self.expansion),
        )

        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.main_path(x) # b, o_c * self.expansion, h / stride, w / stride

        if self.i_downsample: # b, in_c, h, w -> b, o_c, h / stride, w/stride
            x = self.i_downsample(x) 
        
        x += output

        x = self.relu(x)

        return x
        