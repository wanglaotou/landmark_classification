#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :_test22.py
@Date      :2020/03/18 15:13:56
@Author    :mrwang
@Version   :1.0
'''



import torch
import cv2
import sys
import numpy as np
from torchvision import transforms as tf
import data_process
import util
import random
import time
# import dlib
import os
from collections import OrderedDict


#from network import backbone

import loss as ls

# if sys.platform == "win32":
#     sys.path.append("c:/Users/Streamax-JT/Documents/landmark_regression")
#     sys.path.append("C:/code/YoloV2")

# from FaceDetection import faceDetector

# import file_operation as fo
# import pts_operation as po
#import img_and_pts_aug as ipa

# __all__ = ['TestEngineVideo', 'TestEngineImg']
__all__ = ['TestEngineImg']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detected = False
dets = [[0,0,0,0]]
# Detector = faceDetector()

all_type = ['yellow_card Double_column', 'white_card Double_column', 'red_card Double_column', 'green_card Double_column' \
    'yellow_card Single_column', 'white_card Single_column', 'red_card Single_column', 'green_card Single_column']
all_label = [1, 2, 3, 4, 5, 6, 7, 8]
all_dict = {'1': 'yellow_card Double_column', '2':'white_card Double_column', '3':'red_card Double_column', \
    '4':'green_card Double_column', '5':'yellow_card Single_column', '6':'white_card Single_column', \
        '7':'red_card Single_column', '8':'green_card Single_column'}

def draw_points(img,points):

    print('points:', points)
    points *= 128
    
    points = points.reshape((4,2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(points.shape[0]):
        cv2.circle(img,(int(points[i][0]),int(points[i][1])),2,(255,0,0))
        cv2.putText(img,str(i),(int(points[i][0]),int(points[i][1])), font, 0.4, (255, 255, 255), 1)
    return img


class TestEngineImg():
    """ This is a custom engine for this training cycle """

    def __init__(self, modelPath=None, inputSize=[128,128,3]): 
        # print('modepath:',modelPath)

        rs  = data_process.transform.inputResize(inputSize)
        it  = tf.ToTensor()

        self.img_tf = util.Compose([rs, it])
        self.network = torch.load(modelPath)
        self.network = self.network.to(device)
        self.network.eval()

    def __call__(self, imgPathList, inputSize):
        imgList = []
        # print('imgPathList:',imgPathList)

        # ## 3. 直接传入图像img
        # if not isinstance(imgPathList, np.ndarray) or imgPathList.shape[0] <= 0 or imgPathList.shape[1] <= 0:
        #     print("img none!\n")
        # data = self.img_tf(imgPathList)

        # ## 2. 集成车牌检测+车牌点回归
        # if inputSize == [128, 128, 1]:
        #     img = cv2.imread(imgPathList, 0)
        # else:
        #     img = cv2.imread(imgPathList)
        
        # # print(img.shape)
        # if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
        #     print("img none!\n")
        #     # continue
        # #img = img.transpose((2, 0, 1))
        # data = self.img_tf(img)



        # data = torch.unsqueeze(data, 0)
        # # print('data:', data)   # ok
        # data = data.to(device)
        # with torch.no_grad():
        #     out = self.network(data)            

        # eyePoints = out[0].cpu().detach().numpy()
        # return eyePoints


        # 1. 单独测试landmark点回归
        with open(str(imgPathList[0]), 'r') as fi:
            for line in fi:
                line = line.strip().split(' ')
                if len(line) % 8 == 1:
                    imgList.append('/media/mario/新加卷/DataSets/ALPR/zhongdong/' + line[0])
                elif len(line) % 8 == 2:
                    imgList.append('/media/mario/新加卷/DataSets/ALPR/zhongdong/' + line[0] + ' ' + line[1])
                else:
                    continue
        random.shuffle(imgList)
        # print('imgList len:', len(imgList))

        for curPath in imgList:
            print('curPath:', curPath)
            imgpath = ''
            label = ''
            sppath = curPath.strip().split(' ')
            if len(sppath) == 2:
                imgpath = str(sppath[0])
                label = int(sppath[1])
            elif len(sppath) == 3:
                imgpath = str(sppath[0]) + ' ' + str(sppath[1])
                label = int(sppath[2])
            # curPath = 'test_img/5.jpg'
            if inputSize == [128, 128, 1]:
                img = cv2.imread(imgpath, 0)
            else:
                img = cv2.imread(imgpath)
            
            print(img.shape)
            if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
                print("img none!\n")
                # continue
            #img = img.transpose((2, 0, 1))
            data = self.img_tf(img)
            data = torch.unsqueeze(data, 0)
            # print('data:', data)   # ok
            data = data.to(device)
            with torch.no_grad():
                out = self.network(data)
                

            eyePoints = out[0].cpu().detach().numpy()
            pred = out[0].max(1, keepdim=True)[1]
            # return eyePoints
            print('eyePoints, pred, label:', eyePoints, pred, label)
            # colorlan = all_dict[eyePoints]
            # print('colorlan:', colorlan)
            # drawImg = draw_points(img,eyePoints)
            
            # print(label)
            # cv2.namedWindow("img", 0)
            cv2.imshow("img", img)
            cv2.waitKey(0)