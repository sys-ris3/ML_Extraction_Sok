#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

def AlexNet(input_shape, num_classes):
    img_input = layers.Input(shape=input_shape)
    x = Conv2D(96, kernel_size=(11,11), strides= 4,
                    padding= 'valid', activation= 'relu',
                    input_shape= input_shape,
                    kernel_initializer= 'he_normal')(img_input)
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(x)

    x = Conv2D(256, kernel_size=(5,5), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(x)
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(x) 

    x = Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(x)

    x = Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(x)

    x = Conv2D(256, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(x)

    x = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(x)

    x = Flatten()(x)
    x = Dense(4096, activation= 'relu')(x)
    x = Dense(4096, activation= 'relu')(x)
    x = Dense(1000, activation= 'relu')(x)
    x = Dense(num_classes, activation= 'softmax')(x)

    model = training.Model(img_input, x, name='alexnet')
    return model


if __name__ == '__main__':
	model = AlexNet((227, 227, 3), 1000)
	model.summary()
	model.save('alexnet.h5')
