#!/usr/bin/env python
#https://github.com/AmazingGrace-D/SmallVGGNet/blob/master/SmallVGGNet.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation

def MiniVGG(input_shape, classes):
    img_input = layers.Input(shape=input_shape)
    x = Conv2D(32, (3,3), padding= 'same')(img_input)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding= 'same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3,3), padding= 'same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = -1)(x) # Channel last

    x = Conv2D(128, (3,3), padding= 'same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = -1)(x) # Channel last

    x = Conv2D(128, (3,3), padding= 'same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation= 'relu')(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Dropout(0.5)(x)

    x = Dense(classes, activation= 'softmax')(x)

    model = training.Model(img_input, x, name='alexnet')
    return model

if __name__ == '__main__':
    #model = MiniVGG((224, 224, 3), 1000)
    model = MiniVGG((64, 64, 3), 1000)
    model.summary()
    model.save('minivgg.h5')
