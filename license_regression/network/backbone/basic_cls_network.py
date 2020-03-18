#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :basic_cls_network.py
@Date      :2020/03/17 13:44:35
@Author    :mrwang
@Version   :1.0
'''


import os, sys
from collections import OrderedDict
import torch
import torch.nn as nn
from torchstat import stat

# from .. import layer
sys.path.append('/home/mario/Projects/SSD/SSD_mobilenetv2/landmark/license_regression/network')
import layer


class licenseBone(nn.Module):
    def __init__(self):
        super(licenseBone, self).__init__()

        layer0 = layer.Conv2dBatchReLU(3, 32, 3, 2)       ## 采用rgb图像进行训练
        # layer0 = layer.Conv2dBatchReLU(1, 32, 3, 2)         ## 采用灰度图像进行训练
        layer1 = layer.Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = layer.Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = layer.Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = layer.Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

        layer00 = layer.Conv2dBatchReLU(128, 128, 3, 1)
        layer11 = layer.Conv2dBatchReLU(128, 256, 3, 2)
        layer22 = layer.Conv2dBatchReLU(256, 512, 3, 2)
        # layer33 = layer.GlobalAvgPool2d()
        # layer44 = layer.FullyConnectLayer(512, 8)

        self.layers0 = nn.Sequential(
            layer00,
            layer11,
            layer22#,
            # layer33,
            # layer44
        )

        self.gap1 = nn.AvgPool2d(2)
        self.gap2 = layer.GlobalAvgPool2d()
        
        self.fc1 = nn.Linear(2048, 8)    ## landmark
        # self.fc2 = nn.Linear(512, 1)    ## color and lan
        # self.fc1 = layer.FullyConnectLayer(512, 1)
        self.fc2 = layer.FullyConnectLayer(512, 8)

    def forward(self, x):

        x0 = self.layers(x)
        x1 = self.layers0(x0)
        x2 = self.gap1(x1)
        x2 = x2.view(-1, 2048)
        x3 = self.fc1(x2)       ## color and lan

        x4 = self.gap2(x1)
        x5 = self.fc2(x4)       ## landmark

        ## 网络结果预测先出颜色+单双栏，再出点偏移量
        # return x3
        return [x3, x5] 
    
if __name__ == '__main__':  
    model = licenseBone()  
    stat(model,(3, 128, 128))
