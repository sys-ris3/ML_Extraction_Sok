#!/usr/bin/env python
#https://github.com/AmazingGrace-D/SmallVGGNet/blob/master/SmallVGGNet.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Reshape

from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
import random as rand

def get_random_scalar(scalar_stack, isPush):
    if isPush:
        random_scalar = rand.random()
        scalar_stack.append(random_scalar)
    else:
        random_scalar = scalar_stack.pop() * (-1.)
    return random_scalar

def conv(x, filters, kernel, padding, block_id, activation):
    x = Conv2D(filters, 
                kernel_size = kernel, 
                activation = activation,
                padding = padding,
                kernel_initializer = 'he_normal',
                name='conv_%d_obf' % block_id)(x)
    return x

def conv_obf(x, filters, obf_ratio, kernel, padding, block_id, activation, scalar_stack, use_mask=True):
    if (use_mask is True):
        random_scalar = get_random_scalar(scalar_stack, True)
        x = AddMask(random_scalar, name = 'push_mask_%d'%(block_id))(x)

    obf_filters = int(filters*obf_ratio)
    x = Conv2D(obf_filters, 
                kernel_size = kernel, 
                padding = padding,
                kernel_initializer = 'he_normal',
                use_bias=False,
                name='conv_%d_obf' % block_id)(x)
    x = LinearTransform(filters,name="linear_transform_%d"%block_id)(x)
    x = AddMask(1.0, name = 'add_bias_%d'%block_id)(x)

    if (use_mask is True):
        random_scalar = get_random_scalar(scalar_stack, False)
        x = AddMask(random_scalar, name = 'pop_mask_%d'%(block_id))(x)

    x = Activation(activation, name = '%s_%d' % (activation,block_id))(x)
    return x

def MiniVGGObf(input_shape, classes, obf_ratio, scalar_stack):
    # block 1
    img_input = layers.Input(shape=input_shape)
    x = conv_obf(img_input, 32, obf_ratio, (3, 3), 'same', 1, 'relu', scalar_stack, False)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    # block 2 
    x = conv_obf(x, 64, obf_ratio, (3, 3), 'same', 2, 'relu', scalar_stack, True)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    # block 3 
    x = conv_obf(x, 128, obf_ratio, (3, 3), 'same', 3, 'relu', scalar_stack, True)
    x = BatchNormalization(axis = -1)(x) # Channel last

    # block 4 
    x = conv(x, 128, (3, 3), 'same', 4, 'relu')
    x = BatchNormalization(axis = -1)(x) # Channel last

    # block 5 
    x = conv(x, 128, (3, 3), 'same', 5, 'relu')
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    # block 6

    x = Flatten()(x)
    x = Reshape((1,1,8192))(x)

    x = conv(x, 512,(1,1), 'same', 6, 'relu')
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Dropout(0.5)(x)

    # block 7 
    x = conv(x, classes, (1,1), 'same', 7, 'softmax')
    x = Reshape((1000,))(x)

    model = training.Model(img_input, x, name='minivggnet')
    return model

if __name__ == '__main__':
    obf_scalar_stack = []
    model = MiniVGGObf((64, 64, 3), 1000, 1.2, obf_scalar_stack)
    model.summary()
    model.save('minivggobfhh.h5')
