a#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2020/3/15 11:29 AM
# @Author : ChenLiuyin
# @File   : cutsmall.py

import cv2
import os
miu = 200
miut = 150
def cutcr(img):
    img2 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    row, col = img2.shape
    row_e = 0
    col_e = 0
    row_up = 0
    row_down = row - 1
    col_left = 0
    col_right = col - 1
    while True:
        for r in range(row_e, row):
            if img2.sum(axis=1)[r]/row >= miu :
                row_up = r
                break
        for r in range(row - 1, row_e, -1):
            if img2.sum(axis=1)[r]/row >= miu:
                row_down = r
                break
        for c in range(col_e, col):
            if img2.sum(axis=0)[c]/col >= miu :
                col_left = c
                break
        for c in range(col - 1, col_e, -1):
            if img2.sum(axis=0)[c]/col >= miu:
                rcol_right = c
                break
        if img2.sum()/row*col >= miut:
            break
        else:
            row = row_down
            row_e = row_up
            col = col_right
            col_e = col_left

    print(row_up, row_down, col_left, col_right)
    img_cut = img2[row_up:row_down, col_left:col_right]

    return img_cut

#img = cutcr('861433-1_1.jpg')
#cv2.imwrite('cut.jpg', img)

data_dir = '/Users/liuyin/Downloads/new'
cut_dir = '/Users/liuyin/Downloads/cutwhite/cutsmall/second'

classlist = os.listdir(data_dir)
for cla in classlist:
    class_data_dir = os.path.join(data_dir, cla)
    filelist = os.listdir(class_data_dir)
    for file in filelist:
        file_data_dir = os.path.join(class_data_dir, file)
        final_cut_dir = os.path.join(cut_dir, cla, file)
        a = cv2.imread(file_data_dir, cv2.IMREAD_GRAYSCALE)
        try:
            img_cut = cutcr(file_data_dir)
        except:
            print("error")
            print(file_data_dir)
        cv2.imwrite(final_cut_dir, img_cut)

