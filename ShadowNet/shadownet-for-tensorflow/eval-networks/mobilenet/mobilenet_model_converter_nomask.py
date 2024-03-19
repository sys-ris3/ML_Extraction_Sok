#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random

#ModelSafe Obfuscation Related
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel
from model_converter import normal_conv_kernel_obf
from model_converter import convert_conv_layer as convert_conv_layer_custom
from model_converter import generate_dwconv_obf_weight

def convert_pred_layer(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio):
    # get bias from original model
    bias = orig_model.layers[conv_layer].get_weights()[1]

    # convert pred layer like normal conv layer without bias
    convert_conv_layer_custom(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio)

    # set obfuscated mask layer to add bias back
    bias_weight = bias.reshape(1,1,1000)
    obf_model.layers[obf_conv_layer + 2].set_weights([bias_weight])
    return

def convert_conv_layer(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio):
    convert_conv_layer_custom(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio)
    
    # handle BatchNormalization
    bn_ws = orig_model.layers[conv_layer + 1].get_weights()
    obf_model.layers[obf_conv_layer + 2].set_weights(bn_ws)
    return

def convert_dwconv_layer_no_mask(model, layer, obf_model, dw_layer, hasPadding=False):
    if hasPadding:
        pad = 1
    else:
        pad = 0
    dwconv_w = model.layers[layer].get_weights()[0]
    obf_dwconv_w, shuffle_weights, restore_weights = generate_dwconv_obf_weight(dwconv_w)
    obf_model.layers[dw_layer - 1 - pad].set_weights(shuffle_weights)
    obf_model.layers[dw_layer].set_weights([obf_dwconv_w])
    obf_model.layers[dw_layer + 1].set_weights(restore_weights)
    return

def convert_dwconv_layer(model, layer, obf_model, dw_layer, hasPadding):
    convert_dwconv_layer_no_mask(model, layer, obf_model, dw_layer, hasPadding)

    # handle BatchNormalization
    bn_ws = model.layers[layer + 1].get_weights()
    obf_model.layers[dw_layer + 2].set_weights(bn_ws)
    return

def build_layer_id_name_map(model):
    index = None
    layer_to_id={}
    for idx, layer in enumerate(model.layers):
        layer_to_id[layer.name] = idx
    return layer_to_id


"""
MobileNets model converter.
"""
def mobilenet_converter_custom(orig_model, obf_model_tpl, filled_model, obf_ratio):
    # load original model
    model = tf.keras.models.load_model(orig_model)
    model.summary()

    # load template original model
    obf_model = tf.keras.models.load_model(obf_model_tpl, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    obf_model.summary()

    # Prep Conv2D
    convert_conv_layer(model, 2, obf_model, 2, obf_ratio)

    m_layermap = build_layer_id_name_map(model)
    obfm_layermap = build_layer_id_name_map(obf_model)
    for idx in range(1,14):
        layer_name = "conv_dw_%d"%idx
        dwlid = m_layermap[layer_name]
        obf_dwlid = obfm_layermap[layer_name]
        # DepthwiseConv2D
        print("convert %s %d %d"%(layer_name, dwlid, obf_dwlid))
        if idx in [2, 4, 6, 12]:
            convert_dwconv_layer(model, dwlid, obf_model, obf_dwlid, True)
        else:
            convert_dwconv_layer(model, dwlid, obf_model, obf_dwlid, False)


        layer_name = "conv_pw_%d"%idx
        pwlid = m_layermap[layer_name]
        obf_pwlid = obfm_layermap[layer_name]
        # PointwiseConv2D
        print("convert %s %d %d"%(layer_name, pwlid, obf_pwlid))
        convert_conv_layer(model, pwlid, obf_model, obf_pwlid, obf_ratio)

    # Conv Preds 
    convert_pred_layer(model, 90, obf_model, 130, obf_ratio)

    # store converted model in file
    obf_model.save(filled_model)

    return

if __name__ == '__main__':
    orig_model = 'mobilenet.h5'
    obf_custom_tpl = 'mobilenet_obf_nomask.h5'
    filled_model= 'mobilenet_obf_nomask_filled.h5'
    obf_ratio = 1.2
    mobilenet_converter_custom(orig_model, obf_custom_tpl, filled_model, obf_ratio)
