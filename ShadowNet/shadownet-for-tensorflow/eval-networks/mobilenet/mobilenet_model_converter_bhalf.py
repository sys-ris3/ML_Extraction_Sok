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
    if conv_layer == 2: # first conv2d, no input mask
        obf_model.layers[obf_conv_layer + 2].set_weights(bn_ws)
    else:
        obf_model.layers[obf_conv_layer + 3].set_weights(bn_ws)
    return

def convert_dwconv_layer_bhalf(model, layer, obf_model, dw_layer, hasPadding=False):
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
    convert_dwconv_layer_bhalf(model, layer, obf_model, dw_layer, hasPadding)

    # handle BatchNormalization
    bn_ws = model.layers[layer + 1].get_weights()
    obf_model.layers[dw_layer + 3].set_weights(bn_ws)
    return


mask_layer_to_block={5:1,8:1,11:2,12:2, 15:2,18:3,21:3,24:4,25:4,28:4,31:5,34:5,37:6,38:6, 41:6,44:7,47:7,50:8,53:8,56:9,59:9,62:10,65:10,68:11,71:11,74:12,75:12, 78:12,81:13,84:13,90:14}
block_to_scalar = {0:0.01, 1:0.01,2:0.001,3:0.0001,4:0.00001,5:0.0000001,
        6:1.e-07,7:1.e-09,8:1.e-010,9:1.e-012,10:1.e-14, 11:1.e-15,
        12:1.e-17,13:1.e-19,14:1.e-20}

def get_scalar_by_layer(layer):
    block_id = mask_layer_to_block[layer]
    scalar = block_to_scalar[block_id]
    return scalar

"""
Add a encryption/decryption masking layer before/after a given `layer` 
with respect to the original `model` for a `obf_model` to be obfuscated.
"""
def convert_mask_layer(model, layer, obf_model, enc_layer, dec_layer):
    # get and set scaled mask weight
    scalar = get_scalar_by_layer(layer)
    input_mask_weight = obf_model.layers[enc_layer].get_weights()[0]
    input_mask_weights = [scalar * input_mask_weight]
    obf_model.layers[enc_layer].set_weights(input_mask_weights)

    # compute output_mask_weight
    x = tf.convert_to_tensor(input_mask_weights)
    for i in range(enc_layer+1, dec_layer):
        x = obf_model.layers[i](x)
    output_mask_weight = np.array(x)

    # set model with computed output mask weight
    obf_model.layers[dec_layer].set_weights(output_mask_weight)
    return

def copy_model_weights_by_layer(a, b, layers):
    for l in layers:
        weights = a.layers[l].get_weights()
        if weights != None:
            print("copy layer:%d weights"%l)
            b.layers[l].set_weights(weights)
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

    # build layer to id map
    m_layermap = build_layer_id_name_map(model)
    obfm_layermap = build_layer_id_name_map(obf_model)

    # convert block 0 to block 6: layer 0 - 43 
    copy_model_weights_by_layer(model, obf_model, range(44))

    # Convert 7 - 11 layer
    for block_id in range(7, 14):

        print("convert block %d weights" % block_id)
        dwid = m_layermap['conv_dw_%d'%block_id]
        obf_dwid = obfm_layermap['conv_dw_%d'%block_id]
        # DepthwiseConv2D
        if block_id in [2, 4, 6, 12]:
            convert_dwconv_layer(model, dwid, obf_model, obf_dwid, True)
        else:
            convert_dwconv_layer(model, dwid, obf_model, obf_dwid, False)

        mid_push = obfm_layermap['p_push_mask_%d'%(block_id - 1)]
        mid_pop = obfm_layermap['dw_pop_mask_%d'%block_id]
        # Mask for DepthwiseConv2D
        convert_mask_layer(model, dwid, obf_model, mid_push, mid_pop)

        pwid = m_layermap['conv_pw_%d'%block_id]
        obf_pwid = obfm_layermap['conv_pw_%d'%block_id]
        # PointwiseConv2D
        convert_conv_layer(model, pwid, obf_model, obf_pwid, obf_ratio)

        mid_push = obfm_layermap['dw_push_mask_%d'%(block_id)]
        mid_pop = obfm_layermap['pw_pop_mask_%d'%block_id]
        # Mask for PointwiseConv2D
        convert_mask_layer(model, pwid, obf_model, mid_push, mid_pop)

    # Conv Preds 
    # PointwiseConv2D
    pred_lid = obfm_layermap['obf_conv_preds'] 
    convert_pred_layer(model, 90, obf_model, pred_lid, obf_ratio)

    # Mask for PointwiseConv2D
    mid_push = obfm_layermap['tee_conv_preds_push_mask']
    mid_pop = obfm_layermap['tee_conv_preds_pop_mask']
    convert_mask_layer(model, 90, obf_model, mid_push, mid_pop)

    # store converted model in file
    obf_model.save(filled_model)

    return

if __name__ == '__main__':
    orig_model = 'mobilenet.h5'
    obf_custom_tpl = 'mobilenet_obf_bhalf.h5'
    filled_model= 'mobilenet_obf_bhalf_filled.h5'
    obf_ratio = 1.2
    mobilenet_converter_custom(orig_model, obf_custom_tpl, filled_model, obf_ratio)
