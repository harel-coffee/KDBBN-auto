#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2020/3/23 3:08 PM
# @Author : ChenLiuyin
# @File   : findROI.py
import cv2
import numpy as np
from collections import Counter
import os

def custom_threshold(gray):
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:", mean)
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)

    return binary


def exactmask(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # 矩形结构:MORPH_RECT
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 椭圆结构:MORPH_ELLIPSE
    img = cv2.erode(src, kernel)  # 腐蚀
    img = cv2.dilate(img, kernel1)  # 膨胀
    #mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    e_mask = custom_threshold(img)
    return e_mask, img

def inverse(img):
    row, col = img.shape
    # print(row, col)
    iTmp = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            iTmp[i, j] = 255 - img[i, j]
    iTmp = iTmp.astype(int)
    print(iTmp)
    return iTmp

def outlargemask(gray):
    h = 200
    w = 2
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find Contour
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 需要搞一个list给cv2.drawContours()才行！！！！！
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (area < (h / 10 * w / 10)):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        #
        c_max.append(cnt)

    cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)



def outmaxmask(e_mask):
    img = cv2.cvtColor(e_mask, cv2.COLOR_GRAY2BGR)
    # threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    # findContours函数查找图像里的图形轮廓
    # 函数参数thresh是图像对象
    # 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
    # 轮廓逼近方法
    # 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c_max = []
    max_area = 0
    max_cnt = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # find max countour
        if (area > max_area):
            if (max_area != 0):
                c_min = []
                c_min.append(max_cnt)
                #cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)
            max_area = area
            max_cnt = cnt
        else:
            c_min = []
            c_min.append(cnt)
            #cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)

    c_max.append(max_cnt)

    cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def outyuanmask(src):
    maske, img = exactmask(src)
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tre = 115
    while 1:
        ret, thresh = cv2.threshold(img.copy(), tre, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            print(tre, cv2.contourArea(contours[0]))
            break
        else:
            tre = tre-1

    cv2.drawContours(img3, contours, 0, (255, 255, 255), -1)
    img = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img[img != 255] = 0


    ROI = cv2.bitwise_and(img, src)
    return ROI, img3

data_dir = '/Users/liuyin/Downloads/predata/ROI/special'
ROI_dir = '/Users/liuyin/Downloads/predata/ROI/spewithoriROI'
#data_dir = './y2'
#ROI_dir = './y2r'

src = cv2.imread('553894-1_1.jpg', cv2.IMREAD_GRAYSCALE)
maske, img = exactmask(src)
masko = outmaxmask(maske)
#ROI = cv2.bitwise_and(masko, src)
cv2.imwrite('maske.jpg', masko)
'''

classlist = os.listdir(data_dir)
for cla in classlist:
    class_data_dir = os.path.join(data_dir, cla)
    filelist = os.listdir(class_data_dir)
    for file in filelist:
        file_data_dir = os.path.join(class_data_dir, file)
        final_ROI_dir = os.path.join(ROI_dir, cla, file)
        src = cv2.imread(file_data_dir, cv2.IMREAD_GRAYSCALE)
        maske, _ = exactmask(src)
        masko = outmaxmask(maske)
        ROI = cv2.bitwise_and(masko, src)
        #ROI, img3 = outyuanmask(src)
        cv2.imwrite(final_ROI_dir, ROI)
    
filelist = os.listdir(data_dir)
for file in filelist:
    file_data_dir = os.path.join(data_dir, file)
    final_ROI_dir = os.path.join(ROI_dir, file)
    try:
        src = cv2.imread(file_data_dir, cv2.IMREAD_GRAYSCALE)
        #maske = exactmask(src)
        #masko = outmaxmask(maske)
        #ROI = cv2.bitwise_and(masko, src)
        ROI, img3 = outyuanmask(src)
        cv2.imwrite(final_ROI_dir, ROI)
    except:
        print(file)

'''
