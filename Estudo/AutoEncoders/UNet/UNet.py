import torch
from torch import nn
import torchvision

class DoubleConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvLayers, self).__init__()
        
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.first_activation = nn.ReLU()
        
        self.second_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.second_activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        x = self.first_conv(x)
        x = self.first_activation(x)
        x = self.second_conv(x)
        x = self.second_activation(x)
        
        return x
    
    
class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.max_pool(x)
        
        return x
        

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=2,
                                           stride=2)
        
    def forward(self, x):
        x = self.upsample(x)
        
        return x
    
    
class CropAndConcat(nn.Module):
    def __init__(self):
        super(CropAndConcat, self).__init__()
    
    def forward(self, x:torch.Tensor, x_contracting:torch.Tensor):
        
        contracting_x = torchvision.transforms.functional.center_crop(x_contracting, 
                                                                      [x.shape[2], x.shape[3]])
        
        x = torch.cat([x, contracting_x], dim=1)
        
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
              
        self.down_conv = nn.ModuleList([DoubleConvLayers(i, o) for i, o in 
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        
        self.down_max_pool = nn.ModuleList([Down() for _ in range(4)])
        
        self.middle_conv = DoubleConvLayers(512, 1024)
        
        self.up_transpose = nn.ModuleList([Up(i, o) for i, o in 
                                          [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        
        self.up_conv = nn.ModuleList([DoubleConvLayers(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
        self.concat = nn.ModuleList([CropAndConcat() for i in range(4)])
        
    def forward(self, x):
        pass_through = []
        
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)

            pass_through.append(x)
            
            x = self.down_max_pool[i](x)
            
        x = self.middle_conv(x)
        
        for i in range(len(self.up_conv)):
            x = self.up_transpose[i](x)
            
            x = self.concat[i](x, pass_through.pop())
            
            x = self.up_conv[i](x)
            
        x = self.final_conv(x)
        
        return x    
            