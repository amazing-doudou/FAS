import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.onnx as onnx
import math
import os

class HeadBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.branch1_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, 
                                   stride=2, padding=1, groups=1, bias=False)
        self.branch2_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, 
                                   stride=2, padding=1, groups=1, bias=False)
        self.branch2_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, 
                                   stride=2, padding=0, groups=1, bias=False)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, 
                                   stride=1, padding=1, groups=1, bias=False)
        self.inner_IN = nn.InstanceNorm2d(num_features=out_channels//2)
        self.IN = nn.InstanceNorm2d(num_features=out_channels)
        self.nonlinearity = nn.ReLU()
        
    def forward(self, x):
        x1 = self.nonlinearity(self.inner_IN(self.branch1_conv(x)))
        x2 = self.nonlinearity(self.inner_IN(self.branch2_conv1(x) - self.branch2_conv2(x)))
        x = torch.cat([x1, x2], dim=1)
        x = self.IN(self.conv(x))
        return x
        
        
# class EndBlock(nn.Module):
#     def __init__(self, in_channels=256, out_channels=512, task=2):
#         super().__init__()
#         self.aap = nn.AvgPool2d(kernel_size=(4, 4), stride=3, padding=1)
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, 
#                                    stride=1, padding=0, groups=1, bias=False)
#         self.IN = nn.InstanceNorm2d(num_features=out_channels)
#         self.relu = nn.ReLU()
#         self.gap = nn.AdaptiveAvgPool2d(output_size=1)
#         self.linear = nn.Linear(int(out_channels), int(task))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.aap(x)
#         x = self.relu(self.IN(self.conv(x)))
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         x = self.sigmoid(x)
#         return x
        
        
class BaseBlock(nn.Module):
    def __init__(self, kernel_size, input_channels, output_channels, stride, use_se=False):
        super(BaseBlock, self).__init__()
        assert output_channels % 2 == 0
        self.branch1_conv1 = nn.Conv2d(in_channels=input_channels,
                                       out_channels=output_channels // 2,
                                       kernel_size=1,
                                       stride=1,
                                       groups=1,
                                       bias=False)
        self.branch1_conv2 = nn.Conv2d(in_channels=output_channels // 2,
                                       out_channels=output_channels // 2,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=(kernel_size - 1) // 2,
                                       groups=output_channels // 2,
                                       bias=False)
        self.branch1_conv3 = nn.Conv2d(in_channels=output_channels // 2,
                                      out_channels=output_channels // 2,
                                      kernel_size=1,
                                      stride=1,
                                      groups=1,
                                      bias=False)

        self.branch2_conv1 = nn.Conv2d(in_channels=input_channels,
                                       out_channels=output_channels // 2,
                                       kernel_size=1,
                                       stride=1,
                                       groups=1,
                                       bias=False)

        self.branch2_DWconv1 = nn.Conv2d(in_channels=output_channels // 2,
                                         out_channels=output_channels // 2,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=(kernel_size - 1) // 2,
                                         groups=output_channels // 2,
                                         bias=False)
        self.branch2_DWconv2 = nn.Conv2d(in_channels=output_channels // 2,
                                         out_channels=output_channels // 2,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=(kernel_size - 1) // 2,
                                         groups=output_channels // 2,
                                         bias=False)

        self.branch2_conv3 = nn.Conv2d(in_channels=output_channels // 2,
                                       out_channels=output_channels // 2,
                                       kernel_size=1,
                                       stride=1,
                                       groups=1,
                                       bias=False)
        self.branch2_end = nn.Conv2d(in_channels=output_channels,
                                     out_channels=output_channels,
                                     kernel_size=1,
                                     stride=1,
                                     groups=1,
                                     bias=False)
        
        if stride != 1:
            self.downsample = nn.Conv2d(in_channels=input_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size - 1) // 2,
                                        groups=input_channels,
                                        bias=False)
        else:
            self.downsample = None
            
        self.IN = nn.InstanceNorm2d(num_features=output_channels // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x1 = self.IN(self.branch1_conv1(x))
        x1 = self.IN(self.branch1_conv2(x1))
        x1 = self.IN(self.branch1_conv3(x1))

        x2 = self.IN(self.branch2_conv1(x))
        x2_1 = self.branch2_DWconv1(x2)
        x2_2 = self.branch2_DWconv2(x2)
        x2_2 = torch.mul(x2_2, 1)
        x3 = x2_1 - x2_2
        x3 = self.IN(self.branch2_conv3(self.IN(x3)))

        x3 = torch.cat([x1, x3], dim=1)

        x3 = self.IN(self.branch2_end(x3))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(x3 + identity)
        return out


NET_CONFIG = {
    "blocks1": [[3, 128, 128, 2, False], [3, 128, 128, 1, False], [3, 128, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks2": [[3, 128, 256, 2, False], [3, 256, 256, 1, False]]
}


class Liveness(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        self.input_dim = 3
        self.input_size = input_size
        self.blocks0 = nn.Sequential(*[
            HeadBlock(
                in_channels=self.input_dim,
                out_channels=128)
        ])
        # kernel_size, input_channels, output_channels, stride, se = False
        self.blocks1 = nn.Sequential(*[
            BaseBlock(
                kernel_size=k,
                input_channels=in_c,
                output_channels=out_c,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks1"])
        ])
        self.blocks2 = nn.Sequential(*[
            BaseBlock(
                kernel_size=k,
                input_channels=in_c,
                output_channels=out_c,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])
#         self.blocks3 = nn.Sequential(*[
#             EndBlock(
#                 in_channels=256,
#                 out_channels=512,
#                 task=2)
#         ])
        self.output_channel = int(NET_CONFIG["blocks2"][-1][2])
        self._initialize_weights()

    def forward(self, x):
        x = self.blocks0(x)
        out_tir = []
        x = self.blocks1(x)
        out_tir.append(x)
        x = self.blocks2(x)
        out_tir.append(x)
        return out_tir
    
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

                
def Liveness_small(input_size=224):
    model = Liveness(input_size=input_size)
    return model


def export(dir):
    dummy_input = torch.randn(1, 3, 224, 224)
    model = LCNet_small_noSE()
    model.eval()
    torch.save(model.state_dict(), os.path.join(dir, "LCNet_small.pth"))
    onnx.export(model, dummy_input, os.path.join(dir, "LCNet_small.onnx"), verbose=False)
    print('export model successful')

if __name__ == '__main__':
    export('/data/27f7eda9c60443408db9bf86ffcdb265/mmmodel/')