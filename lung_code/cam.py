# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:50:09 2020

@author: sdscit
"""
import numpy as np
import os
import cv2
from keras.models import load_model
from custom_layers.scale_layer import Scale
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

label_dict = {"infil":0,"micro":1,"situ":2}
def read_image(path,label_name):
        img_path = os.listdir(path)
        data = []
        y = np.array([])
        for i in img_path:
            img = cv2.imread(os.path.join(path,i))
            img = cv2.resize(img,(256,256))
            data.append(img)
            y = np.append(y,label_dict[label_name])
        return np.array(data),y
x1,y1 = read_image(r"C:\Users\sdscit\Desktop\all\infil_crf","infil")
x2,y2 = read_image(r"C:\Users\sdscit\Desktop\all\micro_crf","micro")
x3,y3 = read_image(r"C:\Users\sdscit\Desktop\all\situ_crf","situ")
filepath1 =  r"C:\Users\sdscit\Desktop\ct_imgs\model\densenet_169_06-03_all_crf"
model = load_model(filepath1,custom_objects={'Scale': Scale})
HEIGHT = 256
WIDTH = 256
finalconv_name = 'relu5_blk'
fianlconv = model.get_layer(finalconv_name)
weight_softmax = model.layers[-2].get_weights()
def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (WIDTH, HEIGHT)

    # Keras default is channels last, hence nc is in last
    bz, h, w, nc = feature_conv.shape

    output_cam = []
    for idx in class_idx:
        cam = np.dot(weight_softmax[0][:,idx], np.transpose(feature_conv.reshape(h*w, nc)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        output_cam.append(cv2.resize(cam_img, size_upsample))
    
    return output_cam
index = 0
for i in x1[1007:]:
    img_path = os.listdir(r"C:\Users\sdscit\Desktop\all\infil_crf")
    probs_extractor = K.function([model.input], [model.output])
    probs = probs_extractor([np.expand_dims(i, 0)])[0]
    features_conv_extractor = K.function([model.input], [fianlconv.output])
    features_blob = features_conv_extractor([np.expand_dims(i, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    height, width, _ = i.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + i* 0.5
    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\all\infil_cam",img_path[index][:-4]+'_cam.jpg'), result)
    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\all\infil_cam",img_path[index][:-4]+'_heatmap.jpg'), heatmap)
    index+=1
index = 0
for i in x2:
    img_path = os.listdir(r"C:\Users\sdscit\Desktop\all\micro_crf")
    probs_extractor = K.function([model.input], [model.output])
    probs = probs_extractor([np.expand_dims(i, 0)])[0]
    features_conv_extractor = K.function([model.input], [fianlconv.output])
    features_blob = features_conv_extractor([np.expand_dims(i, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    height, width, _ = i.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + i* 0.5
    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\all\micro_cam",img_path[index][:-4]+'_cam.jpg'), result)
    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\all\micro_cam",img_path[index][:-4]+'_heatmap.jpg'), heatmap)
    index+=1
index = 0
for i in x3:
    img_path = os.listdir(r"C:\Users\sdscit\Desktop\all\situ_crf")
    probs_extractor = K.function([model.input], [model.output])
    probs = probs_extractor([np.expand_dims(i, 0)])[0]
    features_conv_extractor = K.function([model.input], [fianlconv.output])
    features_blob = features_conv_extractor([np.expand_dims(i, 0)])[0]
    features_blobs = []
    features_blobs.append(features_blob)
    idx = np.argsort(probs)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
    height, width, _ = i.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + i* 0.5
    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\all\situ_cam",img_path[index][:-4]+'_cam.jpg'), result)
    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\all\situ_cam",img_path[index][:-4]+'_heatmap.jpg'), heatmap)
    index+=1