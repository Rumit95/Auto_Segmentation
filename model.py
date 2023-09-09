import torch
from torch import nn
from torch import tensor
from torchinfo import summary
import torch.nn.functional as F

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Conv2dNormActivation, self).__init__()
        
        if in_channels == out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, exp):
        super(InvertedResidual, self).__init__()
        
        self.residual_connection = True if stride == 1 and in_channels == out_channels else False
        
        if exp == 1:
            self.conv = nn.Sequential(
                Conv2dNormActivation(in_channels*exp, in_channels*exp, kernel_size=3, stride=stride, padding=1, groups=in_channels*exp),
                nn.Conv2d(in_channels*exp, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                    Conv2dNormActivation(in_channels, in_channels*exp, kernel_size=1, stride=1, padding=0, groups=in_channels*exp),
                    Conv2dNormActivation(in_channels*exp, in_channels*exp, kernel_size=3, stride=stride, padding=1, groups=in_channels*exp),
                    nn.Conv2d(in_channels*exp, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        output = self.conv(x)
        if self.residual_connection:
            output = x + self.conv(x)
        return output

class MobileNetV2(nn.Module):
    def __init__(self, input_ch = 3, hidden = 32):
        super(MobileNetV2, self).__init__()
        self.features = self._make_layers(input_ch, hidden)

    def _make_layers(self, input_ch, hidden):
        ###          channels, iteration, stride, expansion
        net_info = [[16, 1, 1, 1],
                    [24, 2, 2, 6],
                    [32, 3, 2, 6],
                    [64, 4, 2, 6],
                    [96, 3, 1, 6],
                    [160, 3, 2, 6],
                    [320, 1, 1, 6]]

        input_channel = hidden
        layers = []
        layers.append(Conv2dNormActivation(input_ch, input_channel, kernel_size=3, stride=2, padding=1, groups=0))

        for channel, itr, stride, expansion in net_info:
            for i in range(itr):
                layers.append(InvertedResidual(in_channels=input_channel, out_channels=channel, exp=expansion, stride=stride))
                input_channel = channel
                stride=1
                
        layers.append(Conv2dNormActivation(320, 1024, kernel_size=1, stride=2, padding=0, groups=0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features[:2](x)
        x2 = self.features[2:4](x1)
        x3 = self.features[4:7](x2)
        x4 = self.features[7:11](x3)
        x5 = self.features[11:14](x4)
        x6 = self.features[14:17](x5)
        x7 = self.features[17:18](x6)
        x8 = self.features[18:](x7)
        return x1, x2, x3, x4, x5, x6, x7, x8
    
class Upsample_Block(nn.Module):
    def __init__(self,in_ch,out_ch,skip_ch,scale):
        super().__init__()
        
        #skip_ch.shape[-1]/in_ch.shape[-1] 
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm2d(num_features=in_ch+skip_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch+skip_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
   

    def forward(self,x,skip):
        #print(x.shape)
        up = self.upsample(x)
        #print(up.shape, skip.shape)
        out = torch.cat((skip,up),axis=1)
        out = self.bn(out)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out

class Modified_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = MobileNetV2(input_ch=3, hidden=24)

        self.dblock1 = Upsample_Block(1024,320,320,2)
        self.dblock2 = Upsample_Block(320,160,160,1)
        self.dblock3 = Upsample_Block(160,96,96,2)
        self.dblock4 = Upsample_Block(96,64,64,1)
        self.dblock5 = Upsample_Block(64,32,32,2)
        self.dblock6 = Upsample_Block(32,24,24,2)
        self.dblock7 = Upsample_Block(24,16,16,2)
        
        self.conv = nn.Conv2d(16,1,kernel_size=1,padding=1)

    def forward(self,x):
        
        d1, d2, d3, d4, d5, d6, d7, d8 = self.encoder(x)
        
        u1 = self.dblock1(d8,d7)
        u2 = self.dblock2(u1,d6) 
        u3 = self.dblock3(u2,d5)
        u4 = self.dblock4(u3,d4)
        u5 = self.dblock5(u4,d3)
        u6 = self.dblock6(u5,d2)
        u7 = self.dblock7(u6,d1)

        out = self.conv(u7)

        return out