#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: mix.py 
@time: 2021/11/20
@software: PyCharm 
'''
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
def sampler(data_dir, p):
    files = []
    labels = []
    for cla in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, cla)):
            files.append(os.path.join(data_dir, cla, file))
            labels.append(cla)
    _, selected, _, _ = train_test_split(files, labels, test_size=p)
    return selected

if __name__ == "__main__":
    data_dir1 = 'C:\\Users\\sdscit\\Desktop\\crf'
    data_dir2 = 'C:\\Users\\sdscit\\Desktop\\black'
    data_dir3 = 'C:\\Users\\sdscit\\Desktop\\white'
    data_dir4 = 'C:\\Users\\sdscit\\Desktop\\miu1'
    data_dir5 = 'C:\\Users\\sdscit\\Desktop\\miu2'
    data_dir6 = 'C:\\Users\\sdscit\\Desktop\\miu3'
    target_dir = 'C:\\Users\\sdscit\\Desktop\\mix'
    beta1, beta2, beta3, beta4, beta5, beta6 = 0.2, 0.2, 0.2, 0.15, 0.15, 0.1
    
    import argparse
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--targetdir', '-t')
    parser.add_argument('--datadir1', '-d1')
    parser.add_argument('--datadir2', '-d2')
    parser.add_argument('--datadir3', '-d3')
    parser.add_argument('--datadir4', '-d4')
    parser.add_argument('--datadir5', '-d5')
    parser.add_argument('--datadir6', '-d6')
    args = parser.parse_args()

    if not os.path.exists(target_dir):
        target_dir = args.targetdir

    if not os.path.exists(data_dir1):
        data_dir1 = args.datadir1

    if not os.path.exists(data_dir2):
        data_dir2 = args.datadir2
        
    if not os.path.exists(data_dir3):
        data_dir3 = args.datadir3
        
    if not os.path.exists(data_dir4):
        data_dir4 = args.datadir4
        
    if not os.path.exists(data_dir5):
        data_dir5 = args.datadir5
        
    if not os.path.exists(data_dir6):
        data_dir6 = args.datadir6


    selected1 = sampler(data_dir1, beta1)
    selected2 = sampler(data_dir2, beta2)
    selected3 = sampler(data_dir3, beta3)
    selected4 = sampler(data_dir4, beta4)
    selected5 = sampler(data_dir5, beta5)
    selected6 = sampler(data_dir6, beta6)
    for file_dir in selected1:
        shutil.copy(file_dir, file_dir.replace(data_dir1, target_dir))

    for file_dir in selected2:
        shutil.copy(file_dir, file_dir.replace(data_dir2, target_dir))

    for file_dir in selected3:
        shutil.copy(file_dir, file_dir.replace(data_dir3, target_dir))

    for file_dir in selected4:
        shutil.copy(file_dir, file_dir.replace(data_dir4, target_dir))

    for file_dir in selected5:
        shutil.copy(file_dir, file_dir.replace(data_dir5, target_dir))

    for file_dir in selected6:
        shutil.copy(file_dir, file_dir.replace(data_dir6, target_dir))
