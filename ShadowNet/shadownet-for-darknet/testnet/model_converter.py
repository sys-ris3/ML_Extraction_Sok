#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel 

"""
Statically assign scalar value for demo, as I haven't
find an easy way to pass parameters from one layer to the following
layers at runtime in tensorflow.

TODO:Will move to real adaptive scaling when implement masking layer
inside TEE.
"""
layer_to_block={1:1,2:2}
block_to_scalar = {1:1.0, 2:.1}
def get_scalar_by_layer(layer):
    block_id = layer_to_block[layer]
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

"""
Obfuscate normal Conv2D weight with given obfuscation ratio.
Note: Assuming weight are stored in channel-last format
@params:
    @weight: original normal Conv2D kernel weight shape of (H,W,C)
    @obf_ratio: (float) equals obfuscated_channels/original_channels
@return: 
    @arr: recording the shuffled sequence of channels
    @obf_dict: recording info for recovering original channel values
    @obf_weight: obfuscated weight, shape of (H, W, C')
"""
def normal_conv_kernel_obf(weight, obf_ratio):
    """
    Naive obfuscation stretagy:
    Assume this Conv2D layer  have N convolution kernels before 
    obfuscation, and we have M kernels after obfuscation.
              M = int(N * obf_ratio)
    Let's take N = 4, obf_ratio = 1.5, then M = 6.
    For simplicity, we assume weight has shape of (H, W, C, N), 
    where H means height, W means width, and C means the number 
    of channels, N means the number of convolution kernels. 
    C depends on the number of channels of input, while N determines
    the number of channels of the output.

    Step1: generate random tensor rand_weight of shape(H, W, C, R)
    Our scheme goes like this, first, we generate random tensor of
    shape (H, W, C, R), R is the number of convolution kernels. 
              R = M - N

    Step2: generate a dictionary records how we obfuscate.
    obf_dict = {orig_chn_id: (obf_chn_id, rand_chn_id, scalar)}
      orig_chn_id: channel id in the weight before obfuscation
      obf_chn_id: channel id in the obf_weight after obfuscation
      rand_chn_id: channel id in the random tensor
      scalar: random scalar used for scaling 

    where:
      weight[h][w][c][orig_chn_id] = 
        obf_weight[h][w][c][obf_chn_id] * scalar
         + rand_weight[h][w][c][rand_chn_id] 

    Step3: generate an empty obf_weight of shape(H, W, C, M)

    Step4: populate the empty obf_weight following the formula:
      obf_weight[h][w][c][obf_chn_id] = 
        (weight[h][w][c][orig_chn_id] - rand_weight[h][w][c][rand_chn_id])
         / scalar
    """

    # get channel number, assuming channel last format
    shape = weight.shape
    chn = shape[-1]

    # calculate target number of channels after obfuscation
    obf_chn = int(chn * obf_ratio)
    obf_shape = shape[:-1] + (obf_chn,)  

    # generate ndarray to store obfuscated weight
    obf_weight = np.ndarray(shape=obf_shape, dtype=float, order='F')

    # generate random part of weight for obfuscation
    rand_weight = tf.random.normal(shape[:-1] + ((obf_chn - chn),))

    # 1. the shuffled location for each channel
    #  channel index of range (0, chn -1) : maps to normal channels 
    #  channel index of range (chn, obf_chn) : maps to random channels
    arr = np.arange(obf_chn)
    np.random.shuffle(arr)

    # 2. generate random scheme 
    obf_dict = {}
    for i in range(chn): # for each original channel
        main_chn = np.where(arr == i)[0][0]
        rand_chn = random.randint(0,obf_chn - chn - 1) + chn
        scalar = random.random()
        obf_dict[i] = (main_chn, rand_chn, scalar)

    # obfuscation
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            for k in range(weight.shape[2]):
                for c in range(obf_chn):
                    # given orig_chn = main_chn * scalar + rand_chn
                    # main_chn = (orig_chn - rand_chn)/scalar
                    if arr[c] < chn: # main_chn
                        rand_chn = obf_dict[arr[c]][1] - chn
                        rand_chn_v = rand_weight[i][j][k][rand_chn]
                        orig_chn_v = weight[i][j][k][arr[c]]
                        scalar = obf_dict[arr[c]][2]
                        main_chn_v = (orig_chn_v - rand_chn_v) / scalar 
                        obf_weight[i][j][k][c] = main_chn_v
                    else: # rand_chn
                        rand_chn = arr[c] - chn
                        rand_chn_v = rand_weight[i][j][k][rand_chn]
                        obf_weight[i][j][k][c] = rand_chn_v

    return arr, obf_dict,obf_weight

"""
@params:
    @m_weight: first part of linear transform layer weight, indexes 
    @s_weight: second part of linear transform layer weight, scalars 
    @shuffle_arr: shuffle array recording sequence of shuffled channel
    @obf_dict: obfuscation dictionary used for recovering original channel
@return:
    No return, modify @weight directly
"""
def populate_linear_transform_weight(m_weight, s_weight, shuffle_arr, obf_dict):
    for i in range(m_weight.shape[1]):
        (main_chn_idx, rand_chn, scalar) = obf_dict[i]
        m_weight[0][i] = main_chn_idx

        rand_chn_idx = np.where(shuffle_arr == rand_chn)[0][0]
        m_weight[1][i] = rand_chn_idx
        s_weight[i] = scalar
    return

def convert_conv_layer(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio):
    weight = orig_model.layers[conv_layer].get_weights()[0]
    arr, obf_dict, obf_weight = normal_conv_kernel_obf(weight, obf_ratio)
    obf_model.layers[obf_conv_layer].set_weights([obf_weight])

    m_weight = obf_model.layers[obf_conv_layer + 1].get_weights()[0]
    s_weight = obf_model.layers[obf_conv_layer + 1].get_weights()[1]
    populate_linear_transform_weight(m_weight, s_weight, arr, obf_dict)
    obf_model.layers[obf_conv_layer + 1].set_weights([m_weight,s_weight])
    return

"""
Assume channel last input format
@params:
    @dwconv_weight:
@return:
    @obf_dwconv_weight
    @dwconv_shuffle_weight
    @dwconv_restore_weight
"""
def generate_dwconv_obf_weight(dwconv_w):
    
    # shuffled arr, take n = 6 for example
    #
    #   0  1  2  3  4  5
    #  -----------------
    # | 5| 2| 1| 0| 3| 4|
    #  -----------------
    #  arr[3] = 0
    # obf_dict[0].idx_to = 3
    # obf_dict[0].idx_from = 5

    # generate shuffled sequence, #channels, 
    # possible dwconv_w.shape (3, 3, 4, 1)
    chn = dwconv_w.shape[-2]
    arr = np.arange(chn)
    np.random.shuffle(arr)

    #dictionary used for 
    obf_dict = {}
    for i in range(chn):
        idx_to = np.where(arr == i)[0][0]
        idx_from = arr[i]
        scalar = random.random()
        obf_dict[i] = (idx_to, idx_from, scalar) 

    # generate pw_conv_shuffle_weight
    obf_shape = (chn,)
    dwconv_shuffle_array = np.ndarray(shape=obf_shape, dtype=int)
    dwconv_shuffle_scalar = np.ndarray(shape=obf_shape, dtype=float, order='F')
    dwconv_restore_array = np.ndarray(shape=obf_shape, dtype=int)
    dwconv_restore_scalar = np.ndarray(shape=obf_shape, dtype=float, order='F')
    for i in range(chn):
        idx_to = obf_dict[i][0]
        idx_from = obf_dict[i][1]
        scalar = obf_dict[i][2]
        dwconv_shuffle_array[i] = idx_from
        dwconv_shuffle_scalar[i] = 1. 
        dwconv_restore_array[i] = idx_to
        dwconv_restore_scalar[i] = 1./scalar 

    shuffle_weights = [dwconv_shuffle_array, dwconv_shuffle_scalar]
    restore_weights = [dwconv_restore_array, dwconv_restore_scalar]

    # generate obfuscated obf_dwconv_weight
    obf_dwconv_w = np.ndarray(shape=dwconv_w.shape, dtype=float, order='F')
    for c in range(dwconv_w.shape[2]):
        idx_from = obf_dict[c][1]
        scalar = obf_dict[idx_from][2]
        for i in range(dwconv_w.shape[0]):
            for j in range(dwconv_w.shape[1]):
                for k in range(dwconv_w.shape[3]):
                    obf_dwconv_w[i][j][c][k] = dwconv_w[i][j][idx_from][k] * scalar
    return obf_dwconv_w, shuffle_weights, restore_weights 

def convert_dwconv_layer(model, layer, obf_model, dw_layer, hasPadding=False):
    pad = 0
    if hasPadding:
        pad = 1
    dwconv_w = model.layers[layer].get_weights()[0]
    obf_dwconv_w, shuffle_weights, restore_weights = generate_dwconv_obf_weight(dwconv_w)
    obf_model.layers[dw_layer].set_weights(shuffle_weights)
    obf_model.layers[dw_layer + pad + 1].set_weights([obf_dwconv_w])
    obf_model.layers[dw_layer + pad + 2].set_weights(restore_weights)
    return

"""
@params
    @orig_model:
    @obf_model_tpl:
    @filled_model
@return
    @None no return, save populated template model in file
"""
def model_converter(orig_model, obf_model_tpl, obf_ratio, filled_model):
    model = tf.keras.models.load_model(orig_model)

    obf_model = tf.keras.models.load_model(obf_model_tpl, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})

    # convert conv layer 
    convert_conv_layer(model, 1, obf_model, 2, obf_ratio)
    
    # convert conv mask layer
    convert_mask_layer(model, 1, obf_model, 1, 4)

    ## convert dwconv layer 
    #convert_dwconv_layer(model, 2, obf_model, 6)

    ## convert dwconv mask layer
    #convert_mask_layer(model, 2, obf_model, 5, 9)

    # store converted model in file
    obf_model.save(filled_model)
    return

if __name__ == '__main__':
    orig_model = 'testnet.h5'
    obf_model_tpl = 'testnet_obf_tpl.h5'
    filled_model= 'testnet_obf_filled.h5'
    obf_ratio = 1.5
    model_converter(orig_model, obf_model_tpl, obf_ratio, filled_model)
