#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME   : 2019/11/20 10:41 PM
# @Author : ChenLiuyin
# @File   : trans2jpg.py
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import cv2
import pandas

def trans(dcm_path, result_path):
    img = sitk.ReadImage(dcm_path)
    image_array = sitk.GetArrayFromImage(img)[0]
    image_array = cv2.resize(image_array, (128, 128))
    image_array = np.array([image_array for i in range(3)]).transpose(1,2,0)
    print(image_array.shape)
    img = image_array.astype(np.float64)
    img = (img-np.min(img))/(np.max(img)-np.min(img))*255
    img = img.astype(int)
    cv2.imwrite(result_path)

file_dir1 = './data_one'
file_dir2 = './data_4in1'
metadata = './dataset1_metadata.csv'
re_dir = './re'

import argparse
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--metadata', '-m')
parser.add_argument('--filedir1', '-f1')
parser.add_argument('--filedir2', '-f2')
parser.add_argument('--redir', '-r')
args = parser.parse_args()

if not os.path.exists(metadata):
    metadata = args.metadata

if not os.path.exists(filedir1):
    file_dir1 = args.filedir1

if not os.path.exists(filedir2):
    file_dir2 = args.filedir2
    
if not os.path.exists(redir):
    re_dir = args.redir
    
df = pd.read_csv(metadata)
for id in os.listdir(file_dir1):
    id_dir = os.path.join(file_dir1, id)
    cla = df.loc[df['patient id']==id].values[0][1]
    cla_dir = os.path.join(re_dir, cla)
    os.makedirs(cla_dir, exist_ok=True)
    for file in os.listdir(id_dir):
        trans(os.path.join(id_dir, file), os.path.join(cla_dir, file))
 
for id in os.listdir(file_dir2):
    id_dir = os.path.join(file_dir2, id)
    cla = df.loc[df['patient id']==id].values[0][1]
    cla_dir = os.path.join(re_dir, cla)
    os.makedirs(cla_dir, exist_ok=True)
    for file in os.listdir(id_dir):
        trans(os.path.join(id_dir, file), os.path.join(cla_dir, file))
