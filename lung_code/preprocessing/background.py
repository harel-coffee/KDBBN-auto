#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2020/3/26 12:32 PM
# @Author : ChenLiuyin
# @File   : changeback.py

import numpy as np
import cv2
import os

def inverse(img):
    iTmp = 255 - img
    #print(iTmp)
    return iTmp

def exactmask(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  
    img = cv2.erode(src, kernel)  
    img = cv2.dilate(img, kernel1) 
    #mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    e_mask = custom_threshold(img)
    return e_mask, img

def outmaxmask(e_mask):
    img = cv2.cvtColor(e_mask, cv2.COLOR_GRAY2BGR)
    
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    
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
                cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)
            max_area = area
            max_cnt = cnt
        else:
            c_min = []
            c_min.append(cnt)
            cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)

    c_max.append(max_cnt)

    cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def custom_threshold(gray):
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:", mean)
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)

    return binary

def change_inwhite(src):
    binary = custom_threshold(src)
    # maske = exactmask(img)
    # mask = outmaxmask(maske)

    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

   
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (255, 255, 55), 5)
    cv2.imwrite('change0.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
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
    print(c_max)

    cv2.drawContours(img, c_max, 0, (25, 255, 255), 5)
    cv2.imwrite('change0.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    img_b = inverse(img)
    change = cv2.bitwise_or(img_b, src)
    return change

def change_special(src, masko):
    img_b = inverse(masko)
    change = cv2.bitwise_or(img_b, src)
    return change

data_dir2 = '/Users/liuyin/Downloads/predata/ROI/black'
data_dir3 = '/Users/liuyin/Downloads/predata/ROI/white'
'''
src = cv2.imread('back5.jpg', cv2.IMREAD_GRAYSCALE)
maske, _ = exactmask(src)
masko = outmaxmask(maske)
change = change_special(src, masko)
cv2.imwrite('changeback.jpg', masko)
'''
classlist = os.listdir(data_dir)
for cla in classlist:
    class_data_dir = os.path.join(data_dir, cla)
    filelist = os.listdir(class_data_dir)
    for file in filelist:
        file_data_dir = os.path.join(class_data_dir, file)
        final_back_dir = os.path.join(back_dir, cla, file)
        src = cv2.imread(file_data_dir, cv2.IMREAD_GRAYSCALE)
        maske, _ = exactmask(src)
        masko = outmaxmask(maske)
        #ROI = cv2.bitwise_and(masko, src)
        change = change_special(src, masko)
        cv2.imwrite(final_back_dir, change)
