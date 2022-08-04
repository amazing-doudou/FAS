# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.onnx as onnx
import torch.nn.functional as F
import math
import os


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=True),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Mbv2_small(nn.Module):
    def __init__(self, input_size=128, width_mult=1.):
        super(Mbv2_small, self).__init__()
        # setting of inverted residual blocks
        '''
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        '''
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 8, 1, 1],
            [6, 16, 2, 2],
            [6, 24, 3, 2],
            [6, 48, 4,  2],
            [6, 64, 3, 1],
            [6, 80, 1, 1],
        ]
        # building first layer
        # print(input_size)
        assert input_size % 32 == 0
        input_channel = int(8 * width_mult)
        self.last_channel = 1
        self.conv1 = conv_bn(3, input_channel, 1)
        self.blocks = []
        # building inverted residual blocks
        for index, (t, c, n, s) in enumerate(self.interverted_residual_setting):
            self.features = []
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
            if index == 0:
                self.block1 = nn.Sequential(*self.features)
            elif index == 1:
                self.block2 = nn.Sequential(*self.features)
            elif index == 2:
                self.block3 = nn.Sequential(*self.features)
            elif index == 3:
                self.block4 = nn.Sequential(*self.features)
            elif index == 4:
                self.block5 = nn.Sequential(*self.features)
            elif index == 5:
                self.block6 = nn.Sequential(*self.features)

        # make it nn.Sequential
        # self.enc = nn.Sequential(*self.features)
        self.output_channel = output_channel
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        out_tir = []
        x = self.block1(x)
        x = self.block2(x)
        out_tir.append(x)
        x = self.block3(x)
        out_tir.append(x)
        x = self.block4(x)
        out_tir.append(x)
        x = self.block5(x)
        out_tir.append(x)
        x = self.block6(x)
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


def MobileNetV2_Small(input_size=128, width_mult=1.,pretrained=False, **kwargs):
    model = Mbv2_small(input_size, width_mult)

    return model


class MobileNetV2_fc(nn.Module):
    def __init__(self, n_class=2, input_size=128, width_mult=1.):
        super(MobileNetV2_fc, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(int(input_size / 32)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )
        self.fc1 = nn.Linear(1280, 2)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        '''
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x,p=0.5,training=self.training)
        '''
        return x

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


def export(dir):
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    model = MobileNetV2_Small()
    model.eval()
    torch.save(model.state_dict(), os.path.join(dir,"MobileNetV2.pth"))
    onnx.export(model, dummy_input, os.path.join(dir,"MobileNetV2.onnx"), verbose=True)


def get_model_and_input(model_save_dir):
    model = MobileNetV2_Small()
    print(model)
    model.cpu()
    model_path = os.path.join(model_save_dir,'MobileNetV2_small.pth')
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    batch_size = 1
    channels = 3
    height = 128
    width = 128
    images = Variable(torch.ones(batch_size, channels, height, width))
    return images, model
def test():
    model = MobileNetV2_Small()
    rgb = torch.rand(2, 3, 128, 128)
    print(model)
    print('size:',model(rgb).size())
    fin_path = '/mnt/sdb4/2d_project/sigle_result/test/checkpoints/MobileNetV2_small.pth'
    torch.save(model.state_dict(), fin_path)
    #print(model)