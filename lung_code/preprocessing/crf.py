# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:29:55 2020

@author: sdscit
"""

import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import os
img_path1 = os.listdir(r"C:\Users\sdscit\Desktop\all\situ")
crf_list = []
for i in img_path1:
    fn_im = os.path.join(r"C:\Users\sdscit\Desktop\all\situ",i)
    fn_anno = os.path.join(r"C:\Users\sdscit\Desktop\all\situ",i)
    fn_output = os.path.join(r"C:\Users\sdscit\Desktop\all\situ_crf",i[:-4]+'_crf.jpg')

    img = cv2.imread(fn_im,0)
    img = cv2.resize(img,(256,256))
    img = np.expand_dims(img, axis=3)
    img = np.concatenate((img, img, img), axis=-1)
    anno_rgb = cv2.resize(cv2.imread(fn_anno),(256,256)).astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    anno_std = np.std(anno_lbl)
    anno_mean = np.mean(anno_lbl)
    anno_median = np.median(anno_lbl)
    anno_percent = np.percentile(anno_lbl, 75)
    for i in range(0,256):
        for j in range(0,256):
            if anno_lbl[i][j]>anno_mean:
                anno_lbl[i][j] = 4227072
            else: anno_lbl[i][j] = 8000000
    colors, labels = np.unique(anno_lbl, return_inverse=True)
    HAS_UNK = 0 in colors

    colorize = np.empty((len(colors), 3), np.uint8)
    
    colorize[:,0] = (colors & 0x0000FF)#得到R层的值，为[0,0], dtype=uint8
    colorize[:,1] = (colors & 0x00FF00) >> 8#得到G层的值，为[ 64, 128], dtype=uint8
    colorize[:,2] = (colors & 0xFF0000) >> 16#得到B层的值，[ 0, 64]

    n_labels = len(set(labels.flat)) - int(HAS_UNK) #返回2，得到除去了label=0后还有两个label
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))


    use_2d = False #是否使用二维指定函数DenseCRF2D，这里设置为False，则说明使用的是一般函数DenseCRF
    #use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        # 创建颜色无关特征，这里只有位置特征，并添加到CRF中
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        # 根据原始图像img创建颜色相关特征和位置相关并添加到CRF中，特征为(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK) #U.shape为(2, 76800),即(n_labels,len(labels))
        d.setUnaryEnergy(U) #将一元势添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2]) #shape为(240, 320)
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        Q = d.inference(5)
        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP,:] #MAP.shape为(76800, 3)，这就是最后的结果
        
    a_ = MAP.reshape(img.shape)
    a = a_.astype(np.uint32)
    b = a[:,:,0] + (a[:,:,1] << 8) + (a[:,:,2] << 16)
    for i in range(0,256):
        if 4227072 in b[i]:
            top = i
            break
    for i in range(255,-1,-1):
        if 4227072 in b[i]:
            down = i
            break
    for i in range(0,256):
        if 4227072 in b[:,i]:
            left = i
            break
    for i in range(255,-1,-1):
        if 4227072 in b[:,i]:
            right = i
            break
    crf_list.append([top,down,left,right])
    '''
    output = cv2.resize(img[top:down+1,left:right+1],(224,224))
    output = cv2.copyMakeBorder(output,16,16,16,16,cv2.BORDER_CONSTANT,value=0)
    cv2.imwrite(fn_output,output)
    '''
gray_values = np.arange(256, dtype=np.uint8)
color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
color_to_gray_map = dict(zip(color_values, gray_values))
img_path2 = os.listdir(r"C:\Users\sdscit\Desktop\all\situ_cam")
group_list = []
for i in img_path2:
        if i[-5] == 'p':
            img_path = os.path.join(r"C:\Users\sdscit\Desktop\all\situ_cam",i)
            img = cv2.imread(img_path)
            img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
            gray_image = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, img)
            avg = 0
            for i in range(0,256):
                for j in range(0,256):
                    avg +=gray_image[i][j]
            avg = avg/(256*256)
            group = []
            for i in range(0,256):
                for j in range(0,256):
                    if gray_image[i][j] > avg*1.2:
                        group.append([i,j])
            group_list.append(group)
#%%

