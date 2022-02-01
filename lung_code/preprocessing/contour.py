#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2020/3/14 4:58 PM
# @Author : ChenLiuyin
# @File   : contour.py

import cv2
import numpy as np

img = cv2.imread("./inversecolor/mask.jpg", cv2.IMREAD_COLOR)
#print(img0.shape)

#img = cv2.pyrDown(cv2.imread("./inversecolor/mask.jpg", cv2.IMREAD_COLOR))
print(img.shape)

# threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
'''
for c in contours:
    # 轮廓绘制方法一
    # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
    x, y, w, h = cv2.boundingRect(c)
    # 在img图像画出矩形，(x, y), (x + w, y + h)是矩形坐标，(0, 255, 0)设置通道颜色，2是设置线条粗度
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

cv2.imwrite("ROI.png", ROI)


'''
# 显示图像
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''

img = cv2.imread('result.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.imshow("img", img)
cv2.waitKey(0)
'''


