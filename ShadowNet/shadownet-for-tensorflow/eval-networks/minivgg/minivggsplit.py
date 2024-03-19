#!/usr/bin/env python
#https://github.com/AmazingGrace-D/SmallVGGNet/blob/master/SmallVGGNet.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Reshape

from tee_shadow_generic_layer import TeeShadowGeneric

def conv_obf(x, h, w, filters, obf_ratio, kernel, padding, block_id):
    obf_filters = int(filters*obf_ratio)
    x = Conv2D(obf_filters, 
                kernel_size = kernel, 
                padding = padding,
                kernel_initializer = 'he_normal',
                use_bias=False,
                name='conv_%d_obf' % block_id)(x)
    if block_id < 7:
        pos = "conv%d" % block_id
    elif block_id == 7:
        pos = "results"
    else:
        print("")
    x = TeeShadowGeneric(h, w, filters, position=pos, name = "ts_conv_%d"%block_id)(x)
    return x

def MiniVGGObf(input_shape, classes, obf_ratio):
    # block 1
    img_input = layers.Input(shape=input_shape)
    x = conv_obf(img_input, 32, 32, 32, obf_ratio, (3, 3), 'same', 1)

    # block 2 
    x = conv_obf(x, 16, 16, 64, obf_ratio, (3, 3), 'same', 2)

    # block 3 
    x = conv_obf(x, 16, 16, 128, obf_ratio, (3, 3), 'same', 3)

    # block 4 
    x = conv_obf(x, 16, 16, 128, obf_ratio, (3, 3), 'same', 4)

    # block 5 
    x = conv_obf(x, 8, 8, 128, obf_ratio, (3, 3), 'same', 5)

    # block 6

    x = Flatten()(x)
    x = Reshape((1,1,8192))(x)

    x = conv_obf(x, 1, 1, 512, obf_ratio, (1,1), 'same', 6)

    # block 7 
    x = conv_obf(x, 1, 1, classes, obf_ratio, (1,1), 'same', 7)
    #x = Reshape((1000,))(x)

    model = training.Model(img_input, x, name='minivggnetsplit')
    return model

if __name__ == '__main__':
    model = MiniVGGObf((64, 64, 3), 1000, 1.2)
    model.summary()
    model.save('minivggsplit.h5')
