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

IMG_H=224
IMG_W=224

if __name__ == '__main__':
    split_model = 'mobilenet_obf_split_filled.h5'
    obf_model= 'mobilenet_obf_custom_filled.h5'
    # load obf model
    o_model = tf.keras.models.load_model(obf_model, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    o_model.summary()

    # load split model 
    s_model = tf.keras.models.load_model(split_model, custom_objects={'TeeShadow':TeeShadow})
    s_model.summary()

    # evaluate the first layer
    print("this layer 0 error:")
    img_input = tf.random.uniform((1,IMG_H, IMG_W, 3))
    obf_results = o_model(img_input)
    s_results = s_model(img_input)
    print("obf_results:%s, s_results:%s"%(obf_results[0][0],s_results[0][0]))
    print("obf_results:%s, s_results:%s"%(obf_results[0][1],s_results[0][1]))
    print("obf_results:%s, s_results:%s"%(obf_results[0][2],s_results[0][2]))
    print("obf_results:%s, s_results:%s"%(obf_results[0][3],s_results[0][3]))
