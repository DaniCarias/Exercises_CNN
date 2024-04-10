from torch import nn
from typing import List
import torch


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        
        self.expansion = 4 # Because in the paper the output channels are 4 times the input channels (60 -> 256 -> ...)
        self.identity_downsample = identity_downsample
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)
        self.bt1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1)
        self.bt2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=1,
                    stride=1,
                    padding=0)
        self.bt3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU()
        
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bt1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bt2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bt3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x
        
        
class ResNetModel(nn.Module):
    def __init__(self, n_layers:List[int], img_channels, num_classes):
        # n_layers -> The number of blocks for each feature map [3, 4, 6, 3]
        super().__init__()
        
        self.expansion = 4 # -> Because in the paper the output channels are 4 times the input channels (60 -> 256 -> ...)
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=64,
                  kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layers(n_layers[0], 64, Block, stride=1)
        self.layer2 = self.make_layers(n_layers[1], 128, Block, stride=2)
        self.layer3 = self.make_layers(n_layers[2], 256, Block, stride=2)
        self.layer4 = self.make_layers(n_layers[3], 512, Block, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * self.expansion, num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        #print(f"After Maxpool {x.shape}")
        
        x = self.layer1(x)
        #print(f"After Layer1 {x.shape}")
        x = self.layer2(x)
        #print(f"After Layer2 {x.shape}")
        x = self.layer3(x)
        #print(f"After Layer3 {x.shape}")
        x = self.layer4(x)
        #print(f"After Layer4 {x.shape}")
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        #print(f"Classification: {x.shape}")
        return x
        
    
    def make_layers(self, n_residual_blocks, intermediate_channels, block, stride):
        layers = []
        
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=intermediate_channels * self.expansion,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2d(intermediate_channels * self.expansion)
        )
        
        layers.append(block(in_channels=self.in_channels, 
                            out_channels=intermediate_channels,
                            identity_downsample=shortcut,
                            stride=stride))
        
        self.in_channels = intermediate_channels * self.expansion
        
        for _ in range(n_residual_blocks - 1):
            layers.append(block(in_channels=self.in_channels, 
                                out_channels=intermediate_channels))
        
        return nn.Sequential(*layers)
    
    
    
    
    
    