#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :dataset11.py
@Date      :2020/03/18 10:05:37
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

__all__ = ['Dataset', 'DatasetWithAngle', 'DatasetWithAngleMulti2']


def get_points(line, start):

    pts = line[start:]
    pts = np.array(pts)
    pts = pts.reshape((8,))

    return pts
    
def read_image_points(path):

    with open(path, "r") as file:

        imgPath = []
        points = []

        for line in file.readlines():

            line = line.strip().split(' ')
            # print('line:', line)
            if len(line) % 8 == 1:
                start = 1
                imgPath.append(line[0])
            elif len(line) % 8 == 2:
                start = 2
                imgPath.append(line[0] + ' ' + line[1])

            points_ = get_points(line, start)
            # print('points:', points_, type(points_))
            # points: ['0.23404255' '0.09433962' '0.95938104' '0.40251572' '0.94197292'
            # '0.91823899' '0.21470019' '0.61635220'] <class 'numpy.ndarray'>
            # points: ['0.18863049' '0.07870370' '0.88630491' '0.00000000' '0.83720930'
            # '0.68055556' '0.15503876' '0.77314815'] <class 'numpy.ndarray'>

            # imgPath.append(line[0])
            points.append(points_)

    return imgPath, points
def sample_image_points(pathList):

    imagepathList = []
    pointsList = []

    shuffleList = []

    newImagepathList = []
    newPointsList = []

    for i in range(len(pathList)):
        curImagepathList, curPointsList = read_image_points(pathList[i])

        imagepathList.extend(curImagepathList)
        pointsList.extend(curPointsList)

    if len(imagepathList) != len(pointsList):
        print("image_smaple has some problem!\n")
        sys.exit()

    for i in range(len(imagepathList)):
        curList = []
        curList.append(imagepathList[i])
        curList.append(pointsList[i])

        shuffleList.append(curList)

    random.shuffle(shuffleList)

    for i in range(len(shuffleList)):

        newImagepathList.append(shuffleList[i][0])
        newPointsList.append(shuffleList[i][1])

    return newImagepathList, newPointsList

class DatasetWithAngleMulti3(torchDataset):

    # imgChannel = 3 对于rgb图像，imgChannel = 1 对于灰度图像
    def __init__(self, imglistPath, inputSize, img_tf, label_tf, imgChannel=1,isTrain='train'):

        if isinstance(imglistPath, (list, tuple)):
            self.imgPathList, self.eyeList = sample_image_points(imglistPath)
        else:
            self.imgPathList, self.eyeList = read_image_points(imglistPath)
        # print(self.imgPathList)
        # print(self.labelList)
        if isTrain == 'train':

            nTrain = int(len(self.imgPathList) * 0.8)

            self.imgPathList = self.imgPathList[0:nTrain]
            self.eyeList = self.eyeList[0:nTrain]

        if isTrain == 'val':

            nTrain = int(len(self.imgPathList) * 0.8)

            self.imgPathList = self.imgPathList[nTrain:]
            self.eyeList = self.eyeList[nTrain:]


        if isTrain == 'trainval':

            self.imgPathList = self.imgPathList
            self.eyeList = self.eyeList

        self.img_tf = img_tf
        self.label_tf = label_tf
        self.num = 0
        self.channel = imgChannel

    def __len__(self):

        return len(self.imgPathList)

    def __getitem__(self, index):

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

                
        return img, self.eyeList[index]

    def collate_fn(self, batch):
        
        images = list()
        eye = list()


        for b in batch:

            images.append(b[0].float())
            eye.append(b[1].tolist())
       
        eye = np.array(eye,dtype=np.float64)

        images = torch.stack(images, dim=0)
        images = torch.FloatTensor(images)
        eye = torch.FloatTensor(eye)

        # print('images, eye:', images.size(), eye.size())
        # images, eye: torch.Size([128, 3, 128, 128]) torch.Size([128, 8])
        return images, eye