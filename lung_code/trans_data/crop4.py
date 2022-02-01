#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2019/11/9 4:06 PM
# @Author : ChenLiuyin
# @File   : crop4.py

import numpy as np
import pydicom
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import os


def crop4(path, filename):
    file_name, d = os.path.splitext(filename)
    img = sitk.ReadImage(os.path.join(path, filename))
    imgarray = sitk.GetArrayFromImage(img)[0]
    w = int(imgarray.shape[0]/2)
    h = int(imgarray.shape[1]/2)
#左上
    new1 = imgarray[0:w, 0:h]
    new1 = new1[np.newaxis, :]
    out1 = sitk.GetImageFromArray(new1)
    result1 = os.path.join(path, file_name+'_1.dcm')
    sitk.WriteImage(out1, result1)
#右上
    new2 = imgarray[w:2*w, 0:h]
    new2 = new2[np.newaxis, :]
    out2 = sitk.GetImageFromArray(new2)
    result2 = os.path.join(path, file_name+'_2.dcm')
    sitk.WriteImage(out2, result2)
#左下
    new3 = imgarray[0:w, h:2*h]
    new3 = new3[np.newaxis, :]
    out3 = sitk.GetImageFromArray(new3)
    result3 = os.path.join(path, file_name+'_3.dcm')
    sitk.WriteImage(out3, result3)
#右下
    new4 = imgarray[w:2*w, h:2*h]
    new4 = new4[np.newaxis, :]
    out4 = sitk.GetImageFromArray(new4)
    result4 = os.path.join(path, file_name+'_4.dcm')
    sitk.WriteImage(out4, result4)

path = "/Users/liuyin/Documents/medicalimage/data/manCT/4in1"
dirlist = os.listdir(path)
for dir in dirlist:
    print(dir)
    try:
        newpath = os.path.join(path, dir)
        imlist = os.listdir(newpath)
        for im in imlist:
            crop4(newpath, im)
    except Exception as ex:
        print("error")
        continue






