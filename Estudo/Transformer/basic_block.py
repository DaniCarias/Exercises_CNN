import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a basic residual block class
class BasicBlock(nn.Module):

    # Set a class-level expansion value for consistent channel behavior
    expansion = 1  # No channel expansion within this block

    # Initialize the block
    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        # Store a flag indicating if this is the last block
        self.is_last = is_last

        # Convolution layers (1st and 2nd) with batch normalization
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Construct the shortcut connection (if needed for dimensional alignment)
        self.shortcut = nn.Sequential()  # Empty by default
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # Forward pass through the block
    def forward(self, x):
        # First convolution, activation, and batch normalization
        out = F.relu(self.bn1(self.conv1(x)))

        # Second convolution and batch normalization
        out = self.bn2(self.conv2(out))

        # Residual addition with the shortcut connection
        out += self.shortcut(x)

        # Store a pre-activation value for later use
        preact = out

        # Final activation before output
        out = F.relu(out)

        # Return both output and pre-activation if last block, otherwise only output
        return out, preact if self.is_last else out
