from torch import nn
from typing import List, Optional



class shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
        
    
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels:int, bottleneck_channels:int, out_channels:int, stride:int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Verify the stride and the number of channels for be the same shape to the next block
        if stride != 1 or in_channels != out_channels:
            self.shortcut = shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity() # No shortcut (Do nothing)
            
        self.act3 = nn.ReLU()
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        x = self.act1(self.bn1(self.conv1(x)))
        
        x = self.act2(self.bn2(self.conv2(x)))
        
        x = self.bn3(self.conv3(x))
        x = self.act3(x + shortcut)
        
        return x
    
class ResNetModel(nn.Module):
    def __init__(self,
                 n_blocks:List[int], # The number of blocks for each feature map/type of convolutions blocks
                 n_channels:List[int], # The number of channels for each layer
                 bootlenecks:List[int], # The number of channels for the bottleneck
                 img_channels:int = 3,
                 first_kernel_size:int = 7, # By default, the first kernel size is 7 in the paper
                 out_features:int = 10 # The number of classes
                ):
        super().__init__()
        
        assert len(n_blocks) == len(n_channels) # Must have the same length because each block have a number of channels
        
        assert len(bootlenecks) == len(n_blocks) 
        
    ### - First Conv Layer - ###
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=n_channels[0], 
                               kernel_size=first_kernel_size, stride=2, padding=first_kernel_size//2)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        
        
        # Create the Residual Blocks
        blocks = []
        
        # To know the number of channels of the previous block -> To check when we need to change the stride
        prev_channel = n_channels[0]
        
        for i, channel in enumerate(n_channels):
            
            # The first conv layer have stride = 2
            stride = 2 if blocks == [] else 1
        
            blocks.append(BottleneckResidualBlock(in_channels=prev_channel, bottleneck_channels=bootlenecks[i],
                                                  out_channels=channel, stride=stride))
            
            prev_channel = channel # Update
            
    ## - Others blocks - ##
            for _ in range(n_blocks[i] - 1):
                 blocks.append(BottleneckResidualBlock(in_channels=channel, bottleneck_channels=bootlenecks[i], 
                                                       out_channels=channel, stride=stride))
        
            self.blocks = nn.Sequential(*blocks) # The * means everything in the blocks list
        
        
        # Classifier
        self.classifier = nn.Sequential(
            #nn.AdaptiveAvgPool2d(output_size=(1,1)), # Global Average Pooling
            nn.Linear(in_features=prev_channel, out_features=out_features)
        )
        
        
    def forward (self, x):
        
    # First Conv Layer
        x = self.bn1(self.conv1(x))

        x = self.blocks(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        
        x = self.classifier(x.mean(dim=-1))

        print(x.shape)
        return x