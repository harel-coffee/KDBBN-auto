#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2019/11/20 10:41 PM
# @Author : ChenLiuyin
# @File   : trans2jpg.py
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import cv2
result_path = '2.jpg'
def trans(dcm_path, result_path):
    img = sitk.ReadImage(dcm_path)
    image_array = sitk.GetArrayFromImage(img)[0]
    image_array = cv2.resize(image_array, (128, 128))
    image_array = np.array([image_array for i in range(3)]).transpose(1,2,0)
    print(image_array.shape)
    img = image_array.astype(np.float64)
    img = (img-np.min(img))/(np.max(img)-np.min(img))*255#*255,归一化
    img = img.astype(int)
    cv2.imwrite(result_path)

file_dir = './data'
re_dir = './re'
for file in os.listdir(file_dir):
    trans(os.path.join(file_dir, file), os.path.join(re_dir, file))
