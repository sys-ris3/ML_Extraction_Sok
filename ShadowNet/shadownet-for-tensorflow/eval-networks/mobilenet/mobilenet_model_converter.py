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
from model_converter import convert_dwconv_layer as convert_dwconv_layer_custom

"""
@params:
    @pw_weight: empty pointwise Conv weight following normal Conv2D layer
    @shuffle_arr: shuffle array recording sequence of shuffled channel
    @obf_dict: obfuscation dictionary used for recovering original channel
@return:
    No return, modify @pw_weight directly
"""
def populate_pw_conv_kernel(pw_weight, shuffle_arr, obf_dict):
    # modify pw_weight directly
    for i in range(pw_weight.shape[0]):
        for j in range(pw_weight.shape[1]):
            for c in range(pw_weight.shape[3]):
                for k in range(pw_weight.shape[2]):
                    pw_weight[i][j][k][c] = 0.
                (main_chn_idx, rand_chn, scalar) = obf_dict[c]
                pw_weight[i][j][main_chn_idx][c] = scalar
                rand_chn_idx = np.where(shuffle_arr == rand_chn)[0][0]
                pw_weight[i][j][rand_chn_idx][c] = 1.0

    return

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

def convert_dwconv_layer(model, layer, obf_model, dw_layer, hasPadding):
    convert_dwconv_layer_custom(model, layer, obf_model, dw_layer, hasPadding)
    pad = 0
    if hasPadding:
        pad = 1

    # handle BatchNormalization
    bn_ws = model.layers[layer + 1].get_weights()
    obf_model.layers[dw_layer + pad + 4].set_weights(bn_ws)
    return

mask_layer_to_block={5:1,8:1,11:2,15:2,18:3,21:3,24:4,28:4,31:5,34:5,37:6,41:6,44:7,47:7,50:8,53:8,56:9,59:9,62:10,65:10,68:11,71:11,74:12,78:12,81:13,84:13,90:14}
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
        b.layers[l].set_weights(weights)
    return

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

    mobilenet_converter_from_loaded_model(model, obf_model, filled_model, obf_ratio)
    return

def mobilenet_converter_from_loaded_model(model, obf_model, filled_model, obf_ratio):
    # Conv Preds 

    # PointwiseConv2D
    convert_pred_layer(model, 90, obf_model, 183, obf_ratio)

    # Mask for PointwiseConv2D
    convert_mask_layer(model, 90, obf_model, 182, 186)

    # Normal Conv2D
    convert_conv_layer(model, 2, obf_model, 2, obf_ratio)

    # Convert 1 - 6 layers
    for block_id in range(1, 7, 2):
        ## Block id  
        p = int((block_id - 1)/2 * 13)
        op = int((block_id - 1)/2 * 27)
        print ("p:%d, op:%d"%(p,op))


        # DepthwiseConv2D
        convert_dwconv_layer(model, 5 + p, obf_model, 7 + op, False)

        # Mask for DepthwiseConv2D
        convert_mask_layer(model, 5 + p, obf_model, 6 + op, 10 + op)

        # PointwiseConv2D
        convert_conv_layer(model, 8 + p, obf_model, 14 + op, obf_ratio)

        # Mask for PointwiseConv2D
        convert_mask_layer(model, 8 + p, obf_model, 13 + op, 16 + op)

        ## Block id + 1 

        # DepthwiseConv2D
        convert_dwconv_layer(model, 12 + p, obf_model, 20 + op, True)

        # Mask for DepthwiseConv2D
        convert_mask_layer(model, 11 + p, obf_model, 19 + op, 24 + op)

        # PointwiseConv2D
        convert_conv_layer(model, 15 + p, obf_model, 28 + op, obf_ratio)

        # Mask for PointwiseConv2D
        convert_mask_layer(model, 15 + p, obf_model, 27 + op, 30 + op)

    # Convert 7 - 11 layer
    for block_id in range(7, 12):
        p = int((block_id - 7) * 6)
        op = int((block_id - 7) * 13)
        print ("p:%d, op:%d"%(p,op))

        # DepthwiseConv2D
        convert_dwconv_layer(model, 44 + p, obf_model, 88 + op, False)

        # Mask for DepthwiseConv2D
        convert_mask_layer(model, 44 + p, obf_model, 87 + op, 91 + op)

        # PointwiseConv2D
        convert_conv_layer(model, 47 + p, obf_model, 95 + op, obf_ratio)

        # Mask for PointwiseConv2D
        convert_mask_layer(model, 47 + p, obf_model, 94 + op, 97 + op)

    ## Block 12 

    # DepthwiseConv2D
    convert_dwconv_layer(model, 75, obf_model, 153, True)

    # Mask for DepthwiseConv2D
    convert_mask_layer(model, 74, obf_model, 152, 157)

    # PointwiseConv2D
    convert_conv_layer(model, 78, obf_model, 161, obf_ratio)

    # Mask for PointwiseConv2D
    convert_mask_layer(model, 78, obf_model, 160, 163)

    ## Block 13 

    # DepthwiseConv2D
    convert_dwconv_layer(model, 81, obf_model, 167, False)

    # Mask for DepthwiseConv2D
    convert_mask_layer(model, 81, obf_model, 166, 170)

    # PointwiseConv2D
    convert_conv_layer(model, 84, obf_model, 174, obf_ratio)

    # Mask for PointwiseConv2D
    convert_mask_layer(model, 84, obf_model, 173, 176)


    # store converted model in file
    obf_model.save(filled_model)

    return



if __name__ == '__main__':
    orig_model = 'mobilenet.h5'
    obf_custom_tpl = 'mobilenet_obf_custom.h5'
    filled_model= 'mobilenet_obf_custom_filled.h5'
    obf_ratio = 1.2
    mobilenet_converter_custom(orig_model, obf_custom_tpl, filled_model, obf_ratio)
