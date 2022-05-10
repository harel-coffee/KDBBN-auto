#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2019/11/7 2:26 PM
# @Author : ChenLiuyin
# @File   : readDCM.py
import os, sys
import pydicom
import SimpleITK as sitk
import skimage.io as io
from matplotlib import pyplot as plt
import numpy as np
import shutil

def del_emp_dir(path):
    for (root, dirs, files) in os.walk(path):
        for item in dirs:
            dir = os.path.join(root, item)
            try:
                os.rmdir(dir)  #os.rmdir() 方法用于删除指定路径的目录。仅当这文件夹是空的才可以, 否则, 抛出OSError。
                print(dir)
            except Exception as e:
                print('Exception',e)

path = "/Users/liuyin/Documents/medicalimage/data/AI-CT/"
resultpath1 = "/Users/liuyin/Documents/medicalimage/data/manCT/one"
resultpath2 = "/Users/liuyin/Documents/medicalimage/data/manCT/4in1"

import argparse
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--path', '-p')
parser.add_argument('--resultpath1', '-r1')
parser.add_argument('--resultpath2', '-r2')
args = parser.parse_args()

if not os.path.exists(path):
    path = args.path

if not os.path.exists(resultpath1):
    resultpath1 = args.resultpath1

if not os.path.exists(resultpath2):
    resultpath2 = args.resultpath2


list = os.listdir(path)
imglist = []

for file in list:
    #img = pydicom.read_file(filename, force=True)
    #img = pydicom.read_file(os.path.join(path, im), force = True)
    newpath = os.path.join(path, file)
    imlist = os.listdir(newpath)
    try:
        os.makedirs(os.path.join(resultpath1, file))
        os.makedirs(os.path.join(resultpath2, file))
    except Exception as ex:
        continue
    for im in imlist:
        try:
            print(os.path.join(newpath, im))
            img = sitk.ReadImage(os.path.join(newpath, im))
    #image_array = np.squeeze(sitk.GetArrayFromImage(img))
            image_array = sitk.GetArrayFromImage(img)
            num = np.array(image_array.shape).size
            if(num==3):
                re = os.path.join(resultpath1, file)
                shutil.copy(os.path.join(newpath, im), os.path.join(re, im))
            else:
                if(num==4):
                    re = os.path.join(resultpath2, file)
                    shutil.copy(os.path.join(newpath, im), os.path.join(re, im))
                else:
                    print("none")
        except Exception as ex:
            print("error")
            continue
    print("over")

del_emp_dir(resultpath1)
del_emp_dir(resultpath2)

