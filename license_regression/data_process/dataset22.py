#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :dataset.py
@Date      :2020/03/17 16:47:05
@Author    :mrwang
@Version   :1.0
'''


import torch
from torch.utils.data.dataset import Dataset as torchDataset
import sys
import cv2
import numpy as np
import random
import time

__all__ = ['Dataset', 'DatasetWithAngle', 'DatasetWithAngleMulti2', 'DatasetWithAngleMulti3']


def get_points(line, start):

    pts = line[start:]
    pts = np.array(pts)
    pts = pts.reshape((8,))

    return pts

def get_colorlan(cl):
    _cl = np.array(cl)
    return _cl

def read_image_points(path):

    with open(path, "r") as file:

        imgPath = []
        colorlan = []
        # points = []
        start = 1
        cl = 1

        for line in file.readlines():

            line = line.strip().split(' ')
            # print('line:', line)
            if len(line) == 2:
                start = 1
                imgPath.append(line[0])
                cl = int(line[1])
                # colorlan.append(cl)
            elif len(line) == 3:
                start = 2
                imgPath.append(line[0] + ' ' + line[1])
                cl = int(line[2])
                # colorlan.append(cl)

            # points_ = get_points(line, start)
            cl_ = get_colorlan(cl)
            # print('cl_:', cl_, type(cl_))
            # cl_: 4 <class 'numpy.ndarray'>
            # cl_: 3 <class 'numpy.ndarray'>
  

            # imgPath.append(line[0])
            # points.append(points_)
            colorlan.append(cl_)

    return imgPath, colorlan#, points
def sample_image_points(pathList):

    imagepathList = []
    colorlanList = []
    # pointsList = []

    shuffleList = []

    newImagepathList = []
    newColorlanList = []
    # newPointsList = []

    for i in range(len(pathList)):
        # curImagepathList, curColorlanList, curPointsList = read_image_points(pathList[i])
        curImagepathList, curColorlanList = read_image_points(pathList[i])

        imagepathList.extend(curImagepathList)
        colorlanList.extend(curColorlanList)
        # pointsList.extend(curPointsList)

    if len(imagepathList) != len(colorlanList):
        print("image_smaple has some problem!\n")
        sys.exit()
    
    # if len(imagepathList) != len(pointsList):
    #     print("image_smaple has some problem!\n")
    #     sys.exit()

    for i in range(len(imagepathList)):
        curList = []
        curList.append(imagepathList[i])
        curList.append(colorlanList[i])
        # curList.append(pointsList[i])

        shuffleList.append(curList)

    random.shuffle(shuffleList)

    for i in range(len(shuffleList)):

        newImagepathList.append(shuffleList[i][0])
        newColorlanList.append(shuffleList[i][1])
        # newPointsList.append(shuffleList[i][2])

    return newImagepathList, newColorlanList#, newPointsList

class DatasetWithAngleMulti3(torchDataset):

    # imgChannel = 3 对于rgb图像，imgChannel = 1 对于灰度图像
    def __init__(self, imglistPath, inputSize, img_tf, label_tf, imgChannel=3, isTrain='train'):

        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList, self.colorlanList = sample_image_points(imglistPath)
        else:
            self.imgPathList, self.colorlanList = read_image_points(imglistPath)
        # print(self.imgPathList)
        # print(self.labelList)
        if isTrain == 'train':

            nTrain = int(len(self.imgPathList) * 0.8)

            self.imgPathList = self.imgPathList[0:nTrain]
            self.colorlanList = self.colorlanList[0:nTrain]
            # self.eyeList = self.eyeList[0:nTrain]

        if isTrain == 'val':

            nTrain = int(len(self.imgPathList) * 0.8)

            self.imgPathList = self.imgPathList[nTrain:]
            self.colorlanList = self.colorlanList[nTrain:]
            # self.eyeList = self.eyeList[nTrain:]


        if isTrain == 'trainval':

            self.imgPathList = self.imgPathList
            self.colorlanList = self.colorlanList
            # self.eyeList = self.eyeList

        self.img_tf = img_tf
        self.label_tf = label_tf
        self.num = 0
        self.channel = imgChannel

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):
        
        # print('self.imgPathList[index]:', self.imgPathList[index])
        if(self.channel == 1):

            img = cv2.imread("/media/mario/新加卷/DataSets/ALPR/zhongdong/" + self.imgPathList[index], 0)

        else:
            img = cv2.imread("/media/mario/新加卷/DataSets/ALPR/zhongdong/" + self.imgPathList[index])

        if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
            print("img none! and is {}\n".format(self.imgPathList[index]))
            sys.exit()

        if self.img_tf is not None:
            # print(type(img))
            # print(img.shape)
            # img = img[np.newaxis,:,:]
            # print(img.shape)
            img = self.img_tf(img)

        # print('img, self.colorlanList[index]:', img.shape, self.colorlanList[index])
        # img, self.colorlanList[index]: torch.Size([3, 128, 128]) 5
        # img, self.colorlanList[index]: torch.Size([3, 128, 128]) 8
        return img, self.colorlanList[index]#, self.eyeList[index]

    def collate_fn(self, batch):
        
        images = list()
        vllabel = list()
        # eye = list()
        # label = int(annotation[1])

        # print('batch size:', batch)
        for b in batch:
            # print('b:', b[0], b[1], type(b[0]), type(b[1]))
            images.append(b[0].float())
            vllabel.append(b[1].tolist())
            # eye.append(b[2].tolist())
       
        vllabel = np.array(vllabel, dtype=np.float64)
        # eye = np.array(eye, dtype=np.float64)

        images = torch.stack(images, dim=0)
        images = torch.FloatTensor(images)
        vllabel = torch.FloatTensor(vllabel)
        # eye = torch.FloatTensor(eye)

        # print('images, vllabel:', images.size(), vllabel.size())
        # images, vllabel: torch.Size([128, 3, 128, 128]) torch.Size([128])
        return images, vllabel#, eye