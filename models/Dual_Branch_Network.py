import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.onnx as onnx
import torch.nn.functional as F

import os
import math
from models.CDCNs import CDCN, Conv2d_cd
from models.LCNet import LCNet_small
from models.attentions import DANetHead
from models.unet_parts import Up

class CDCN_Encoder(nn.Module):
    def __init__(self, last_channel=1):
        super(CDCN_Encoder, self).__init__()
        basic_conv=Conv2d_cd
        theta = 0.7
        self.channels = [128, 128, 128, 128]
        self.last_channel = last_channel
        self.encode = CDCN(basic_conv)
        self.dan1 = DANetHead(128, 64)
        self.dan2 = DANetHead(128, 64)
        self.dan3 = DANetHead(128, 64)
        
        self.downsample14x14 = nn.Upsample(size=(14, 14), mode='bilinear')
        self.lastconv1 = nn.Sequential(
            basic_conv(64*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, last_channel, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.encode(x)
        x1 = self.dan1(out[-3])[0]
        x1 = self.downsample14x14(x1)
        x2 = self.dan1(out[-2])[0]
        x2 = self.downsample14x14(x2)
        x3 = self.dan1(out[-1])[0]
        x3 = self.downsample14x14(x3)
        x_concat = torch.cat((x1,x2,x3), dim=1)
        last = self.lastconv1(x_concat) # 1 1 14 14 
        return out, last
    
class LCNet_Encoder(nn.Module):
    def __init__(self, last_channel=160):
        super(LCNet_Encoder, self).__init__()
        basic_conv=Conv2d_cd
        theta = 0.7
        self.channels = [16*2, 24*2, 48*2, 64*2, 80*2]
        self.last_channel = last_channel
        self.encode = LCNet_small() 
        self.dan1 = DANetHead(48*2, 64)
        self.dan2 = DANetHead(64*2, 64)
        self.dan3 = DANetHead(80*2, 64)
        self.lastconv1 = nn.Sequential(
            basic_conv(64*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, last_channel, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        self.downsample14x14 = nn.Upsample(size=(14, 14), mode='bilinear')
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.encode(x)
        x1 = self.dan1(out[-3])[0]
        x1 = self.downsample14x14(x1)
        x2 = self.dan2(out[-2])[0]
        x2 = self.downsample14x14(x2)
        x3 = self.dan3(out[-1])[0]
        x3 = self.downsample14x14(x3)
        x_concat = torch.cat((x1,x2,x3), dim=1)
        last = self.lastconv1(x_concat) # 1 1 14 14 
        return out, last

class Decoder(nn.Module):
    def __init__(self, channels, last_channel, bilinear=True):
        super(Decoder, self).__init__()
        basic_conv=Conv2d_cd
        theta = 0.7
        print('channels:', channels)
        self.up1 = Up(last_channel, channels[-1], bilinear)
        self.up2 = Up(channels[-1], channels[-2], bilinear)
        self.up3 = Up(channels[-2], channels[-3], bilinear)
        self.up4 = Up(channels[-3], channels[-4], bilinear)
        self.lastconv1 = nn.Sequential(
            basic_conv(channels[-4], channels[-4]*2, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(channels[-4]*2),
            nn.ReLU(),
            basic_conv(channels[-4]*2, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
    def forward(self, out, last): 
#         out, last = self.encode(x)
        x1 = self.up1(last, out[-1])
        x2 = self.up2(x1, out[-2])
        x3 = self.up3(x2, out[-3])
        x4 = self.up4(x3, out[-4])
        out = self.lastconv1(x4)
        return out
        
    
def export(dir):
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    model = LCNet_Encoder()
    model.eval()
    torch.save(model.state_dict(), os.path.join(dir,"LCNet_small.pth"))
    onnx.export(model, dummy_input, os.path.join(dir,"LCNet_small.onnx"), verbose=True)
    
def test_Encoder():
    model = LCNet_Encoder()
    rgb = torch.rand(1,  3, 224, 224)
    print(model)
    print('size:', model(rgb)[1].size())

def test():
    encode = LCNet_Encoder()
    model = Decoder(encode)
    rgb = torch.rand(1,  3, 224, 224)
    print(model)
    print('size:', model(rgb).size())
    
if __name__ == '__main__':
    test()