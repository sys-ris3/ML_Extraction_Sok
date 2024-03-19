#!/usr/bin/env python
import datetime
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Layer,Conv2D,Dense,InputLayer,AveragePooling2D,MaxPooling2D,Activation,Flatten,Reshape,Dropout,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,Lambda,DepthwiseConv2D,ReLU, Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.nn import relu6

from alexnet import AlexNet
import cv2
import urllib
import requests
import numpy as np
from bs4 import BeautifulSoup
import os

# quantized layer for any non-linear computation
class ActivationQ(Layer):

    def __init__(self, activation, bits_w, bits_x, quantize=True,
                 **kwargs):
        super(ActivationQ, self).__init__(**kwargs)
        self.bits_w = bits_w 
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.quantize = quantize
        self.activation = activation
        assert activation in ["relu", "relu6", "softmax"]

    def activation_name(self):
        return self.activation

    def call(self, inputs):
        #inputs = tf.Print(inputs, [tf.reduce_sum(tf.abs(tf.cast(inputs, tf.float64)))], message="relu input: ")
        #inputs = tf.Print(inputs, [], message="in ActivationQ with input shape: {}".format(inputs.get_shape().as_list()))

        if self.quantize:
            inputs_dq = inputs/(self.range_x * self.range_w)
            if self.activation in ["relu", "relu6"]:
                if self.activation.endswith("relu6"):
                    act = relu(inputs, max_value=6 * self.range_x * self.range_w)
                else:
                    act = relu(inputs)

                outputs = tf.math.round(act / self.range_w)
            else: # softmax
                outputs = tf.nn.softmax(inputs_dq)
        else:
            if self.activation == "relu":
                outputs = tf.nn.relu(inputs)
            elif self.activation == "relu6":
                outputs = tf.nn.relu6(inputs)
            else:
                outputs = tf.nn.softmax(inputs)
        return outputs 

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'activation': self.activation,
        }
        base_config = super(ActivationQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def create_dir_safely(directory):
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

def train(model_path):
    EPOCHS = 100
    BATCH_SIZE = 32
    image_height = 227
    image_width = 227
    train_dir = "/root/mobilenet-eval/dataset/mini-imagenet/train_1"
    valid_dir = "/root/mobilenet-eval/dataset/mini-imagenet/val_1"
    model_dir = model_path
    num_classes = 4 
    model = AlexNet((image_height,image_width,3), num_classes)
    model.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
                      rescale=1./255,
                      rotation_range=10,
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      shear_range=0.1,
                      zoom_range=0.1)
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")
    
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    train_num = train_generator.samples
    valid_num = valid_generator.samples
    model.fit(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=train_num // BATCH_SIZE,
                    validation_data=valid_generator,
                    validation_steps=valid_num // BATCH_SIZE)
    model.save(model_dir, include_optimizer=False)
    return

def compare(m_ori, m_obf, original_quantize=True, obf_model_not_quantize=False):
    BATCH_SIZE = 32
    image_height = 227
    image_width = 227
    valid_dir = "/root/mobilenet-eval/dataset/mini-imagenet/val_1"
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    #model_ori = tf.keras.models.load_model(m_ori)
    model_ori = tf.keras.models.load_model(m_ori, custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})
    model_obf = tf.keras.models.load_model(m_obf, custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})
    valid_num = valid_generator.samples
    match_ori  = 0
    match_obf  = 0
    match_con  = 0
    total = 0
    for i in range(valid_num//BATCH_SIZE):
        x,y = valid_generator.__getitem__(i)
        x_q = x.astype('float32')*256.0
        if original_quantize:
            ori = model_ori.predict(x_q)
        else:
            ori = model_ori.predict(x)
        if obf_model_not_quantize:
            obf = model_obf.predict(x)
        else:
            obf = model_obf.predict(x_q)
        
        for a in range(len(y)):
            max_y_idx = np.argmax(y[a], axis=0) 
            max_ori_idx = np.argmax(ori[a], axis=0) 
            max_obf_idx = np.argmax(obf[a], axis=0) 
            if (max_y_idx == max_ori_idx):
                match_ori += 1 
            if (max_y_idx == max_obf_idx):
                match_obf += 1 
            if (max_ori_idx == max_obf_idx):
                match_con += 1 
            total += 1 
    print("totol: %d, match_obf: %d, match_ori: %d, match_con: %d" % (total, match_obf, match_ori, match_con))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='alexnet_train_and_eval')
    parser.add_argument('-t', '--train', action='store_true',
            help = 'train model or evaluate model')
    parser.add_argument('modelpath',
            help = 'path to save model file')
    parser.add_argument('-c', '--compare-model', default = 'trained_alexnet.h5',
            help = 'the path to the model to be compared with')
    parser.add_argument('-q', '--quantize-model', action='store_true',
            help = 'comparing model is also quantized')
    parser.add_argument('-n', '--no-quantize', action='store_true',
            help = 'obf model is also not quantized')
    args = parser.parse_args()
    orig_model = args.modelpath
    if args.train:
        train(args.modelpath)
    else:
        compare(args.compare_model, args.modelpath, args.quantize_model, args.no_quantize)
