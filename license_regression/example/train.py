#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :train.py
@Date      :2020/03/18 10:01:39
@Author    :mrwang
@Version   :1.0
'''


import sys
sys.path.insert(0, '.')
sys.path.append("/home/mario/Projects/SSD/SSD_mobilenetv2/landmark/license_regression")
import engine


if __name__ == '__main__':

    '''
        ## 执行顺序
        1. 先训练点回归网络，test_colorlan_lmark11.txt
        2. 再训练分类网络，test_colorlan_lmark22.txt
        3. 最后训练点回归+分类网络，test_colorlan_lmark33.txt   
    '''
    samplePath5 = "/media/mario/新加卷/DataSets/ALPR/zhongdong/lmark_colorlan/trainval.txt"
    model_Path = None
    samplePathList = []
    ## 设置训练采用rgb图像还是灰度图像，True表示为rgb图像，False表示灰度图像
    train_rgb = True
    inputSize=[]
    imgChannel = 1
    if train_rgb:
        inputSize=[128, 128, 3]
        imgChannel = 3
    else:
        inputSize=[128, 128, 1]
        imgChannel = 1

    samplePathList.append(samplePath5)

    eng = engine.TrainingEngine(modelPath=model_Path ,imgListPath=samplePathList, classNum=8, batchSize=128, workers=8, imgChannel = imgChannel, inputSize=inputSize)#
    eng()