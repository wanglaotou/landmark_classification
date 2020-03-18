'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 15:06:07
@Description: 
'''
import sys
import random
sys.path.insert(0, '.')
sys.path.append("/home/mario/Projects/SSD/SSD_mobilenetv2/landmark/license_regression")
import engine
               

if __name__ == '__main__':

    if True:              
 
        modelPath = "/home/mario/Projects/SSD/SSD_mobilenetv2/landmark/license_regression/models/model_lmark_colorlan/multi_epoch_49_lmark_cl.pkl"
               
        files = ['/media/mario/新加卷/DataSets/ALPR/zhongdong/lmark_colorlan/test_colorlan_lmark33.txt']

        train_rgb = True
        inputSize=[]

        if train_rgb:
            inputSize=[128, 128, 3]
            imgChannel = 3
        else:
            inputSize=[128, 128, 1]
            imgChannel = 1

        eng = engine.TestEngineImg(modelPath=modelPath, inputSize=inputSize)
       
        eng(files, inputSize) 
                                                    