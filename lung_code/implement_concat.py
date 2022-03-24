# -*- coding: utf-8 -*-


from keras.optimizers import SGD, Adam
from keras.layers import Input, merge, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from custom_layers.scale_layer import Scale
from sklearn.metrics import log_loss, accuracy_score, recall_score
from keras.models import load_model
import numpy as np
import os
import cv2
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf
from keras import losses
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def densenet169_model(img_input1, img_input2, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    eps = 1.1e-5
    global concat_axis
    concat_axis = 3
    # compute compression factor
    compression = 1.0 - reduction

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169)
    img_input = Input(shape=(256, 256, 3))
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    model = Model(img_input,x)

    final_stage = stage + 1
    x_new1 = model(img_input1)
    x_new1, nb_filter = dense_block(x_new1, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    x_new1 = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x_new1)
    x_new1 = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x_new1)
    x_new1 = Activation('relu', name='relu'+str(final_stage)+'_blk')(x_new1)
    x_new1 = GlobalAveragePooling2D(name='pool'+str(final_stage))(x_new1)
    x_new1 = Dense(num_classes, name='fc6_1')(x_new1)
    return x_new1


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:

            nb_filter += growth_rate
    return concat_feat, nb_filter

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def res_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    res_conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_input1, img_input2, color_type=1, num_classes=None):
    """
    Resnet 50 Model for Keras

    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    # if K.image_dim_ordering() == 'tf':
    # if K.image_data_format() == 'channels_first':
    bn_axis = 3
    '''
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))
      '''

    x = ZeroPadding2D((3, 3))(img_input2)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = res_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = res_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = res_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = res_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input2, x_fc)


    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)


    return x_newfc


def class_weighted_crossentropy(target, output):
    axis = -1
    output /= tf.reduce_sum(output, axis, True)
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.2, 0.4, 0.4]
    return -tf.reduce_sum(target * weights * tf.log(output))

#%%
def read_image(path,label_name):
    img_path = os.listdir(path)
    data = []
    y = np.array([])
    for i in img_path:
        img = cv2.imread(os.path.join(path,i),0)
        img = cv2.resize(img,(256,256))
        data.append(img)
        y = np.append(y,label_dict[label_name])
    return np.array(data),y

def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

if __name__ == '__main__':

    re_dir = './re'
    target_dir = './mix'
    TRAIN = True
    img_rows, img_cols = 256, 256# Resolution of inputs
    channel = 3
    num_classes = 3
    batch_size = 16
    nb_epoch = 100
    alpha = 0.6
    label_dict = {"IAC":0,"MIA":1,"AIS":2}

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


    # representation data reading
    x1,y1 = read_image(os.path.join(re_dir, 'IAC'), "IAC")
    x2,y2 = read_image(os.path.join(re_dir, 'MIA'), "MIA")
    x3,y3 = read_image(os.path.join(re_dir, 'AIS'), "AIS")
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=20)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=20)
    x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=20)
    x_train = np.concatenate((x1_train,x2_train,x3_train))
    y_train  = np.concatenate((y1_train,y2_train,y3_train))
    x_test = np.concatenate((x1_test,x2_test,x3_test))
    y_test  = np.concatenate((y1_test,y2_test,y3_test))
    # rebalance data reading
    x1m,y1m = read_image(os.path.join(target_dir, 'IAC'), "IAC")
    x2m,y2m = read_image(os.path.join(target_dir, 'MIA'), "MIA")
    x3m,y3m = read_image(os.path.join(target_dir, 'AIS'), "AIS")
    x1_trainm, x1_testm, y1_trainm, y1_testm = train_test_split(x1m, y1m, test_size=0.3, random_state=20)
    x2_trainm, x2_testm, y2_trainm, y2_testm = train_test_split(x2m, y2m, test_size=0.3, random_state=20)
    x3_trainm, x3_testm, y3_trainm, y3_testm = train_test_split(x3m, y3m, test_size=0.3, random_state=20)
    x_trainm = np.concatenate((x1_train, x2_train, x3_train))
    y_trainm = np.concatenate((y1_train, y2_train, y3_train))
#%%
    img_input1 = Input(shape=(256, 256, 3))
    img_input2 = Input(shape=(256, 256, 3))
    output1 = densenet169_model(img_input1 ,img_input2, color_type=channel, num_classes=num_classes)
    output2 = resnet50_model(img_input1 ,img_input2, color_type=channel, num_classes=num_classes)
    filepath = r"C:\Users\sdscit\Desktop\ct_imgs\model\merge"+time.strftime("%m-%d",time.localtime())+"-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    output1 = Activation('softmax', name='prob_1')(output1)
    output2 = Activation('softmax', name='prob_2')(output2)
    # rebalance data resampling
    x_train_ = []
    for i in range(0,1696):
        x_train_.append(x_trainm[i].flatten())
    x_train_ = np.array(x_train_)
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.datasets import make_imbalance
    rand = RandomOverSampler(random_state=10)
    feature_r, label_r = rand.fit_sample(x_train_, y_train)
    x_res, y_res = make_imbalance(feature_r, label_r,
                                  sampling_strategy={0: 566, 1: 565, 2: 565},
                                  random_state=10)
    x_res = np.reshape(x_res,(1696,256,256))
    y_train = np_utils.to_categorical(y_train,num_classes=3)
    y_test = np_utils.to_categorical(y_test,num_classes=3)
    y_res = np_utils.to_categorical(y_res,num_classes=3)
    x_train = np.expand_dims(x_train, axis=3)
    x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
    x_res = np.expand_dims(x_res, axis=3)
    x_res = np.concatenate((x_res, x_res, x_res), axis=-1)    
    x_test = np.expand_dims(x_test, axis=3)
    x_test = np.concatenate((x_test, x_test, x_test), axis=-1)
    x_train_gen = datagen.flow(x_train, batch_size=batch_size)
    x_res_gen = datagen.flow(x_res, batch_size=batch_size)
    x_test_gen = datagen.flow(x_test, batch_size=batch_size)
    model = Model(inputs=[img_input1,img_input2],outputs=[output1,output2])
    if TRAIN:
        # train
        model.compile(optimizer=sgd, loss=[losses.categorical_crossentropy,losses.categorical_crossentropy], loss_weights=[0.5,0.5], metrics=['accuracy'])
        model.fit_generator([x_train_gen,x_res_gen],
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      shuffle=True,
                      verbose=2,
                      validation_data=([x_test_gen,x_test_gen]),
                      callbacks = callbacks_list
                      )
        K.clear_session()
        tf.reset_default_graph()

    else:
        # predict
        modelpath = os.listdir(r'C:\Users\sdscit\Desktop\ct_imgs\model')
        for i in modelpath:
            if i[-4:] == 'hdf5':
                model = model.load_model(os.path.join(r'C:\Users\sdscit\Desktop\ct_imgs\model',i),custom_objects={'Scale': Scale})
                model1 = Model(inputs=model.input[0],outputs=model.get_layer('fc6_1').output)
                model2 = Model(inputs=model.input[1],outputs=model.get_layer('fc10').output)
                feature1 = model1.predict(x_test)
                feature2 = model2.predict(x_test)
                feature = feature1*alpha+feature2*(1-alpha)
                y_pre = softmax(feature)
            

