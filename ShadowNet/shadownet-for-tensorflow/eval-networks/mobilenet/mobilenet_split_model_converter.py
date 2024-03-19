#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random

#ModelSafe Obfuscation Related
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel
from tee_shadow_layer import TeeShadow 


def fill_split_model(obf_model, split_model, layer_mapping, filled_split_model):
    for o_layer in layer_mapping:
        s_layer = layer_mapping[o_layer]
        ws = obf_model.layers[o_layer].get_weights()
        split_model.layers[s_layer].set_weights(ws)
    split_model.save(filled_split_model)


if __name__ == '__main__':
    split_model = 'mobilenet_obf_split.h5'
    filled_split_model = 'mobilenet_obf_split_filled.h5'
    obf_model= 'mobilenet_obf_custom_filled.h5'
    # load obf model
    o_model = tf.keras.models.load_model(obf_model, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    o_model.summary()

    # load split model 
    s_model = tf.keras.models.load_model(split_model, custom_objects={'TeeShadow':TeeShadow})
    s_model.summary()

    layer_mapping = {2:2,8:4,14:6, 22:9, 28:11,35:13,41:15,49:18,55:20,62:22,68:24,76:27,82:29,89:31,95:33,102:35,108:37,115:39,121:41,128:43,134:45,141:47,147:49, 155:52,161:54,168:56, 174:58, 183:60}

    fill_split_model(o_model, s_model, layer_mapping, filled_split_model)
