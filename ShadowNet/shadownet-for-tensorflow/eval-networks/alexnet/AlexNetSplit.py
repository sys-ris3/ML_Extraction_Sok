#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Reshape

from tee_shadow_generic_layer import TeeShadowGeneric
import random as rand

def conv_obf(x, h, w, filters, obf_ratio, kernel, strides, padding, block_id):
    obf_filters = int(filters*obf_ratio)
    x = Conv2D(obf_filters, 
                kernel_size = kernel, 
                strides = strides,
                padding = padding,
                kernel_initializer = 'he_normal',
                use_bias=False,
                name='conv_%d_obf' % block_id)(x)
    if block_id < 9:
        pos = "conv%d" % block_id
    elif block_id == 9: # last block, use 'results'
        pos = "results"
    else:
        print("Error in conv_obf: block_id illegal!")

    x = TeeShadowGeneric(h, w, filters, position=pos, name = "ts_conv_%d"%block_id)(x) 
    return x

def AlexNetObf(input_shape, num_classes, obf_ratio, scalar_stack):
    img_input = layers.Input(shape=input_shape)
    x = conv_obf(img_input, 27, 27, 96, obf_ratio, (11, 11), 4, 'valid', 1)
    x = conv_obf(x, 13, 13, 256, obf_ratio, (5,5), 1, 'same', 2)
    x = conv_obf(x, 13, 13, 384, obf_ratio, (3,3), 1, 'same', 3)
    x = conv_obf(x, 13, 13, 384, obf_ratio, (3,3), 1, 'same', 4) 
    x = conv_obf(x, 6, 6, 256, obf_ratio, (3,3), 1, 'same', 5)
    x = Flatten()(x)
    x = Reshape((1,1,9216))(x)
    x = conv_obf(x, 1, 1, 4096, obf_ratio, (1,1), 1,'same', 6)
    x = conv_obf(x, 1, 1, 4096, obf_ratio, (1,1), 1,'same', 7)
    x = conv_obf(x, 1, 1, 1000, obf_ratio, (1,1), 1,'same', 8)
    x = conv_obf(x, 1, 1, num_classes, obf_ratio, (1,1),1,'same',9)
    #x = Reshape((1000,))(x)

    model = training.Model(img_input, x, name='alexnetsplit')
    return model

if __name__ == '__main__':
    obf_scalar_stack = []
    model = AlexNetObf((227, 227, 3), 1000, 1.2, obf_scalar_stack)
    model.summary()
    model.save('alexnetsplit.h5')
