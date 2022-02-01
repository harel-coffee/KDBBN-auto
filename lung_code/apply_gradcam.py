import numpy as np
import os
import cv2
from keras.models import load_model
from keras.applications import imagenet_utils
from custom_layers.scale_layer import Scale
from pyimagesearch.gradcam import GradCAM
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
from sklearn.cluster import KMeans
#tf.compat.v1.disable_eager_execution()

label_dict = {"IAC":0,"MIA":1,"AIS":2}
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
x1,y1 = read_image(r"C:\Users\sdscit\Desktop\allROI\IAC","IAC")
x2,y2 = read_image(r"C:\Users\sdscit\Desktop\allROI\MIA","MIA")
x3,y3 = read_image(r"C:\Users\sdscit\Desktop\allROI\AIS","AIS")
filepath1 =  r"C:\Users\sdscit\Desktop\ct_imgs\model\densenet_169_04-02_allROI_revise"
model = load_model(filepath1,custom_objects={'Scale': Scale})
for image in x1:
    img_path = os.listdir(r"C:\Users\sdscit\Desktop\allROI\IAC")
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    i = np.argmax(preds[0])
    cam = GradCAM(model, i,'conv5_32_x2')
    heatmap = cam.compute_heatmap(image)
    (heatmap, output) = cam.overlay_heatmap(heatmap, image, alpha=0.5)
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    output = np.vstack([image, heatmap, output])
    cv2.imshow("Output", output)
    cv2.waitKey(0)

#for i in x2:
#    img_path = os.listdir(r"C:\Users\sdscit\Desktop\allROI\MIA")
#    probs_extractor = K.function([model.input], [model.output])
#    probs = probs_extractor([np.expand_dims(i, 0)])[0]
#    features_conv_extractor = K.function([model.input], [fianlconv.output])
#    features_blob = features_conv_extractor([np.expand_dims(i, 0)])[0]
#    features_blobs = []
#    features_blobs.append(features_blob)
#    idx = np.argsort(probs)
#    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
#    height, width, _ = i.shape
#    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
#    result = heatmap * 0.3 + i* 0.5
#    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\allROI\MIA_cam",img_path[index][:-4]+'_cam.jpg'), result)
#    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\allROI\MIA_cam",img_path[index][:-4]+'_heatmap.jpg'), heatmap)
#    index+=1
#index = 0
#for i in x3:
#    img_path = os.listdir(r"C:\Users\sdscit\Desktop\allROI\AIS")
#    probs_extractor = K.function([model.input], [model.output])
#    probs = probs_extractor([np.expand_dims(i, 0)])[0]
#    features_conv_extractor = K.function([model.input], [fianlconv.output])
#    features_blob = features_conv_extractor([np.expand_dims(i, 0)])[0]
#    features_blobs = []
#    features_blobs.append(features_blob)
#    idx = np.argsort(probs)
#    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])
#    height, width, _ = i.shape
#    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
#    result = heatmap * 0.3 + i* 0.5
#    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\allROI\AIS_cam",img_path[index][:-4]+'_cam.jpg'), result)
#    cv2.imwrite(os.path.join(r"C:\Users\sdscit\Desktop\allROI\AIS_cam",img_path[index][:-4]+'_heatmap.jpg'), heatmap)
#    index+=1
