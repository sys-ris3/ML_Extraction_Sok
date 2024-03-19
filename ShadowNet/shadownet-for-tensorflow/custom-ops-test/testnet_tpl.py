#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras import backend
# from tensorflow.python.keras import layers
from keras import backend,layers
# from tensorflow.keras.applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape

from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel
import random as rand

NUM_OBF_FILTERS = 6 
NUM_FILTERS = 4 
IMG_H =  224
IMG_W =  224 
obf_scalar_stack = []

def get_random_scalar(scalar_stack, isPush):
    if isPush:
        random_scalar = rand.random()
        scalar_stack.append(random_scalar)
    else:
        random_scalar = scalar_stack.pop() * (-1.)
    return random_scalar

def TestNetObfuscated(input_shape=None,
        filters=NUM_FILTERS,
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
    random_scalar = get_random_scalar(obf_scalar_stack, True)
    x = AddMask(random_scalar, name="conv_push_mask")(img_input)
    print("push mask shape")
    print(x.shape)
    x = layers.Conv2D(
        NUM_OBF_FILTERS,
        kernel,
        padding='valid',
        use_bias=False,
        strides=strides,
        name='conv1_obf')(
            x)
    x = LinearTransform(NUM_FILTERS, name="linear_transform")(x)
    random_scalar = get_random_scalar(obf_scalar_stack, False)
    x = AddMask(random_scalar, name="conv_pop_mask")(x)
    random_scalar = get_random_scalar(obf_scalar_stack, True)
    x = AddMask(random_scalar, name="dwconv_push_mask")(x)
    x = ShuffleChannel(name="dwconv_shuffle_channel")(x)
    x = layers.DepthwiseConv2D((3, 3),
                        padding='valid',
                        use_bias=False,
                        strides=strides,
                        name='conv_dw')(
                            x)
    x = ShuffleChannel(name="dwconv_restore_channel")(x)
    random_scalar = get_random_scalar(obf_scalar_stack, False)
    x = AddMask(random_scalar, name="dwconv_pop_mask")(x)
    model = keras.Model(inputs=img_input, outputs=x, name='testnet_obf_tpl')
    return model

if __name__ == '__main__':
    img_inputs = keras.Input(shape=(IMG_H,IMG_W,3))
    model = TestNetObfuscated(input_tensor=img_inputs)
    model.summary()
    model.save('testnet_obf_tpl.h5')
