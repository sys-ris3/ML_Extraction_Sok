#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Reshape

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
    
def conv(x, filters, kernel, strides, padding, block_id, activation):
    x = Conv2D(filters, 
                kernel_size = kernel, 
                strides = strides,
                activation = activation,
                padding = padding,
                kernel_initializer = 'he_normal',
                name='conv_%d_obf' % block_id)(x)
    return x

def conv_obf(x, filters, obf_ratio, kernel, strides, padding, block_id, activation, scalar_stack, use_mask=True):
    if (use_mask is True):
        random_scalar = get_random_scalar(scalar_stack, True)
        x = AddMask(random_scalar, name = 'push_mask_%d'%(block_id))(x)

    obf_filters = int(filters*obf_ratio)
    x = Conv2D(obf_filters, 
                kernel_size = kernel, 
                strides = strides,
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

def AlexNetObf(input_shape, num_classes, obf_ratio, scalar_stack):
    img_input = layers.Input(shape=input_shape)
    x = conv(img_input, 96, (11, 11), 4, 'valid', 1, 'relu')
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(x)

    x = conv(x, 256, (5,5), 1, 'same', 2, 'relu')
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(x) 

    x = conv(x, 384, (3,3), 1, 'same', 3, 'relu')
    x = conv(x, 384, (3,3), 1, 'same', 4, 'relu') 
    x = conv(x, 256, (3,3), 1, 'same', 5, 'relu')
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(x)
    x = Flatten()(x)
    x = Reshape((1,1,9216))(x)

    x = conv_obf(x, 4096, obf_ratio, (1,1), 1,'same', 6, 'relu', scalar_stack, True)
    x = conv_obf(x, 4096, obf_ratio, (1,1), 1,'same', 7, 'relu', scalar_stack, True)
    x = conv_obf(x, 1000, obf_ratio, (1,1), 1,'same', 8, 'relu', scalar_stack, True)
    x = conv_obf(x, num_classes, obf_ratio, (1,1),1,'same',9,'softmax',scalar_stack, True)
    x = Reshape((1000,))(x)

    model = training.Model(img_input, x, name='alexnetobf')
    return model

if __name__ == '__main__':
    obf_scalar_stack = []
    model = AlexNetObf((227, 227, 3), 1000, 1.2, obf_scalar_stack)
    model.summary()
    model.save('alexnetobfbh.h5')
