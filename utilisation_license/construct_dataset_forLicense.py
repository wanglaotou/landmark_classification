#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :construct_dataset_forLicense.py
@Date      :2020/03/17 13:46:36
@Author    :mrwang
@Version   :1.0
'''


import glob
import cv2
import os, sys
import numpy as np 
import pickle
import copy
import time
import random
import xml.dom.minidom
from dataAug_pts import *

# color type: ['yellow_card', 'white_card', 'red_card', 'green_card'] 4
# lan type: ['Double_column', 'Single_column'] 2

ideal_type = ['yellow_card Double_column', 'white_card Double_column', 'red_card Double_column', 'green_card Double_column' \
    'yellow_card Single_column', 'white_card Single_column', 'red_card Single_column', 'green_card Single_column']


all_type = ['yellow_card Double_column', 'white_card Single_column', 'red_card Double_column', 'white_card Double_column' \
    'green_card Double_column', 'red_card Single_column', 'green_card Single_column', 'yellow_card Single_column']
all_label = [0, 1, 2, 3, 4, 5, 6, 7]   # 标签从0开始

root_dir = '/media/mario/新加卷/DataSets/ALPR/zhongdong'
img_dir = os.path.join(root_dir, 'check_tag')

save_dir = "lmark_colorlan/lmark_cl_train"
pos_save_dir = os.path.join(root_dir,save_dir)

if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)

f1 = open(os.path.join(root_dir, 'lmark_colorlan/train.txt'), 'w')

with open(os.path.join(root_dir, 'lmark_colorlan/zd_trainval_lmark_cl.txt') ,'r') as f:
    lines = f.readlines()

# 统计真正干净的数据
valid_data = 0
for line in lines:
    vidall = ''
    annotations = line.strip().split(' ')
    if len(annotations) % 14 == 2:
        jpgfile = annotations[0]
        # print('jpgfile:', jpgfile)
        filename = os.path.basename(jpgfile)
        filename = os.path.splitext(filename)[0]
        
        img_ori = cv2.imread(os.path.join(img_dir,jpgfile))
        height, width, channel = img_ori.shape
        n = int(annotations[1])
        vidall = str(annotations[2]) + ' ' + str(annotations[3])
        points = []

        if n == 1:
            for i in range(8):
                # if i < 4:
                #     points.append(int(annotations[4+i]))
                # if i >= 4:
                    # points.append(float(annotations[4+i]))
                
                points.append(float(annotations[4+i]))

            points = np.array(points)
            # pts_ori = points.reshape((6,2))
            pts_ori = points.reshape((4,2))
            
            # boxes = pts_ori[0:2]
            # pts_ori = pts_ori[2:]
          
        else:
            continue

    if len(annotations) % 14 == 3:
        jpgfile = annotations[0] + ' ' + annotations[1]
        # print(jpgfile)
        filename = os.path.basename(jpgfile)
        filename = os.path.splitext(filename)[0]
        
        img_ori = cv2.imread(os.path.join(img_dir,jpgfile))
        height, width, channel = img_ori.shape
        n = int(annotations[2])
        vidall = str(annotations[3]) + ' ' + str(annotations[4])
        points = []

        if n == 1:
            for i in range(8):
                # if i < 4:
                #     points.append(int(annotations[5+i]))
                # if i >= 4:
                #     points.append(float(annotations[5+i]))
                points.append(float(annotations[5+i]))

            points = np.array(points)
            # pts_ori = points.reshape((6,2))
            pts_ori = points.reshape((4,2))
            
            # boxes = pts_ori[0:2]
            # pts_ori = pts_ori[2:]
        
        
        else:
            continue

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # for i in range (4):
    #     cv2.circle(img_ori,(int(pts_ori[i][0]),int(pts_ori[i][1])),3,(0,255,0),-1)
    #     cv2.putText(img_ori,str(i),(int(pts_ori[i][0]),int(pts_ori[i][1])), font, 0.4, (255, 255, 255), 1)
    # cv2.imshow('img',img_ori)
    # cv2.waitKey(0)
    
    # 生成正样本
    count = 0
    pos_num = 1
    total_each_num = 1
    colorlanlabel = ''
    if vidall == 'yellow_card Double_column':
        total_each_num = 10
        colorlanlabel = str(0)
    elif vidall == 'white_card Single_column':
        total_each_num = 1 
        colorlanlabel = str(1)
    elif vidall == 'red_card Double_column':
        total_each_num = 2
        colorlanlabel = str(2)
    elif vidall == 'white_card Double_column':
        total_each_num = 2
        colorlanlabel = str(3)
    elif vidall == 'green_card Double_column':
        total_each_num = 10
        colorlanlabel = str(4)
    elif vidall == 'red_card Single_column':
        total_each_num = 2
        colorlanlabel = str(5)
    elif vidall == 'green_card Single_column':
        total_each_num = 10
        colorlanlabel = str(6)
    elif vidall == 'yellow_card Single_column':
        total_each_num = 20
        colorlanlabel = str(7)
    
    while pos_num < total_each_num:
        count += 1
        if count > 100000:
            break
        
        img = copy.deepcopy(img_ori)
        pts = copy.deepcopy(pts_ori)

        img,pts = randomAug(img,pts)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # for i in range (4):
        #     cv2.circle(img,(int(pts[i][0]),int(pts[i][1])),3,(0,255,0),-1)
        #     cv2.putText(img,str(i),(int(pts[i][0]),int(pts[i][1])), font, 0.4, (255, 255, 255), 1)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        h,w = img.shape[0:2]
        
        # print(pts[:,0])
        pts[:,0] = pts[:,0] / float(w)
        pts[:,1] = pts[:,1] / float(h)


        resized_im = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        save_file = os.path.join(pos_save_dir, "{}_pos_{}.jpg".format(filename,pos_num))
        cv2.imwrite(save_file, resized_im)

        f1.write(save_file + ' ')

        for i in range(pts.shape[0]):
            x = pts[i][0]
            y = pts[i][1]
            f1.write('%.8f %.8f ' %(x, y))
        f1.write(colorlanlabel + '\n')
        # f1.write('\n')

        pos_num +=1
        
    valid_data += 1
    # time.sleep(10)
    print("{} images done, pos: {}".format(valid_data, pos_num))

f1.close()



