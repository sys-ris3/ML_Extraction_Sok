#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from keras import backend, layers
# from tensorflow.python.keras import backend
# from tensorflow.python.keras import layers
# from tensorflow.keras.applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape

NUM_FILTERS = 4
IMG_H = 8
IMG_W = 8

"""
 Simple TestNet model with one typical layers
"""
def TestNet(input_shape=None,
        filters= NUM_FILTERS,
        weights=None,
        include_top=True,
        input_tensor=None):

    default_size = IMG_H 
    kernel = (3,3)
    strides = (1,1)

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=NUM_FILTERS,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)
    
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        filters,
        kernel,
        padding='valid',
        use_bias=False,
        strides=strides,
        name='conv1')(
            img_input)
    x = layers.DepthwiseConv2D((3, 3),
                        padding='valid',
                        use_bias=False,
                        strides=strides,
                        name='conv_dw')(
                            x)
    model = keras.Model(inputs=img_input, outputs=x, name='testnet_model')
    return model

if __name__ == '__main__':
    img_inputs = keras.Input(shape=(IMG_H,IMG_W,3))
    model = TestNet(input_tensor=img_inputs)
    model.summary()
    model.save('testnet.h5')
