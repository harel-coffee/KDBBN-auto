#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2020/3/14 4:58 PM
# @Author : ChenLiuyin
# @File   : contour.py

import cv2
import numpy as np


def SROI(img, roi_path):
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
    print(img.shape)
    src = cv2.imread('553894-1_1.jpg', cv2.IMREAD_GRAYSCALE)
    ROI = cv2.bitwise_and(img, src)

    cv2.imwrite(roi_path, ROI)

data_dir2 = './black'
re_dir = './re'

import argparse
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--redir', '-r')
parser.add_argument('--datadir2', '-d')
args = parser.parse_args()

if not os.path.exists(re_dir):
    re_dir = args.redir

if not os.path.exists(data_dir2):
    data_dir2 = args.datadir2

    
for cla in os.listdir(re_dir):
    os.makedirs(os.path.join(data_dir2, cla), exist_ok=True)
    for file in os.listdir(os.path.join(re_dir, cla)
        img = cv2.imread(os.path.join(re_dir, cla, file), cv2.IMREAD_COLOR)
        SROI(img, os.path.join(data_dir2, cla, file))

