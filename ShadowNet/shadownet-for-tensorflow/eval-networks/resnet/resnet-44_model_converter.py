#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from IPython import embed
import tensorflow as tf
import numpy as np
import random
import os
from time import time
#ModelSafe Obfuscation Related
from add_mask_layer import AddMask
#from linear_transform_layer import LinearTransform
from linear_transform_generic_layer import LinearTransformGeneric
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

def convert_conv_layer(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio,stacked = False,batch_norm = True):
    convert_conv_layer_custom(orig_model, conv_layer, obf_model, obf_conv_layer, obf_ratio,stacked)
    
    # handle BatchNormalization
    
    if batch_norm:
        if conv_layer == 1: # first conv2d, no input mask
            obf_bn_layer = obf_model.layers[obf_conv_layer + 2]
            orig_bn_layer = orig_model.layers[conv_layer + 1]
        else:
            if not stacked:
                obf_bn_layer = obf_model.layers[obf_conv_layer + 3]
                orig_bn_layer = orig_model.layers[conv_layer + 1]
            else:
                obf_bn_layer = obf_model.layers[obf_conv_layer + 6]
                orig_bn_layer = orig_model.layers[conv_layer + 2]
        assert('batch_normalization' in orig_bn_layer.name)
        bn_ws = orig_bn_layer.get_weights()
        assert('batch_normalization' in obf_bn_layer.name)
        obf_bn_layer.set_weights(bn_ws)
    return


mask_layer_to_block={5 : 1,
                     8 : 1,
                     11: 2,
                     15: 2,
                     18: 3,
                     21: 3,
                     24: 4,28:4,31:5,34:5,37:6,41:6,44:7,47:7,50:8,53:8,56:9,59:9,62:10,65:10,68:11,71:11,74:12,78:12,81:13,84:13,90:14}
block_to_scalar = {0 : 0.01, 
                   1 : 0.01,
                   2 : 0.001,
                   3 : 0.0001,
                   4 : 0.00001,
                   5 : 0.0000001,
        6:1.e-07,7:1.e-09,8:1.e-010,9:1.e-012,10:1.e-14, 11:1.e-15,
        12:1.e-17,13:1.e-19,14:1.e-20}

def get_scalar_by_layer(layer):
    # block_id = mask_layer_to_block[layer]
    # scalar = block_to_scalar[block_id]
    return 1.0

"""
Add a encryption/decryption masking layer before/after a given `layer` 
with respect to the original `model` for a `obf_model` to be obfuscated.
"""
def convert_mask_layer(model, layer, obf_model, enc_layer, dec_layer,stacked = False):
    # get and set scaled mask weight
    scalar = get_scalar_by_layer(layer)

    obf_push_mask_layer = obf_model.layers[enc_layer]
    assert('push_mask' in obf_push_mask_layer.name)
    input_mask_weight = obf_push_mask_layer.get_weights()[0]
    input_mask_weights = [scalar * input_mask_weight]
    obf_push_mask_layer.set_weights(input_mask_weights)

    # compute output_mask_weight
    x = tf.convert_to_tensor(input_mask_weights)
    if stacked:
        for i in range(enc_layer+2, dec_layer,2):
            x = obf_model.layers[i](x)
        output_mask_weight = np.array(x)
    else:
        for i in range(enc_layer+1, dec_layer):
            x = obf_model.layers[i](x)
        output_mask_weight = np.array(x)

    # set model with computed output mask weight

    obf_pop_mask_layer = obf_model.layers[dec_layer]
    assert('pop_mask' in obf_pop_mask_layer.name)

    obf_pop_mask_layer.set_weights(output_mask_weight)
    return

def copy_model_weights_by_layer(a, b, layers):
    for l in layers:
        weights = a.layers[l].get_weights()
        b.layers[l].set_weights(weights)
    return

def resnet_convert_conv_block_normal(model,original_layer_counter,obf_model,obf_layer_counter,obf_ratio):
    convert_conv_layer(model, original_layer_counter ,obf_model,obf_layer_counter,obf_ratio)
    convert_mask_layer(model,original_layer_counter,obf_model,obf_layer_counter - 1,obf_layer_counter + 2)
    original_layer_counter += 3
    obf_layer_counter += 6
    
    convert_conv_layer(model, original_layer_counter ,obf_model,obf_layer_counter,obf_ratio)
    convert_mask_layer(model,original_layer_counter,obf_model,obf_layer_counter - 1,obf_layer_counter + 2)
    original_layer_counter += 4
    obf_layer_counter += 7

    return original_layer_counter,obf_layer_counter

def resnet_convert_conv_block_complex(model,original_layer_counter,obf_model,obf_layer_counter,obf_ratio):
    
    convert_conv_layer(model, original_layer_counter ,obf_model,obf_layer_counter,obf_ratio)
    convert_mask_layer(model,original_layer_counter,obf_model,obf_layer_counter - 1,obf_layer_counter + 2)
    
    original_layer_counter += 3
    obf_layer_counter += 7
    
    convert_conv_layer(model, original_layer_counter,obf_model,obf_layer_counter, obf_ratio,stacked = True,batch_norm=True)
    convert_mask_layer(model, original_layer_counter,obf_model,obf_layer_counter - 2,obf_layer_counter + 4,stacked = True)

    original_layer_counter += 1
    obf_layer_counter += 1
    
    convert_conv_layer(model, original_layer_counter,obf_model,obf_layer_counter, obf_ratio,stacked=True,batch_norm=False)
    convert_mask_layer(model, original_layer_counter,obf_model,obf_layer_counter - 2,obf_layer_counter + 4,stacked = True)

    original_layer_counter += 4
    obf_layer_counter += 9

    return original_layer_counter,obf_layer_counter
"""
resnets model converter.
"""
def resnet_converter_custom(orig_model, obf_model_tpl, filled_model, obf_ratio):
    # load original model
    model = tf.keras.models.load_model(orig_model)
    ## model.summary()

    # load template original model
    obf_model = tf.keras.models.load_model(obf_model_tpl, custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ShuffleChannel':ShuffleChannel})
    ## obf_model.summary()

    resnet_converter_from_loaded_model(model, obf_model, filled_model, obf_ratio)
    return


def resnet_converter_from_loaded_model(model, obf_model, filled_model, obf_ratio):
    
    depth = 44

    num_res_blocks = int((depth - 2) / 6)

    original_layer_counter = 1
    obf_layer_counter = 1

    convert_conv_layer(model, original_layer_counter ,obf_model,obf_layer_counter,obf_ratio)
    original_layer_counter += 3
    obf_layer_counter += 5

    for stack in range(3):
        for res_block in range(num_res_blocks):
            time1 = time()
            if stack > 0 and res_block == 0:
                original_layer_counter,obf_layer_counter = resnet_convert_conv_block_complex(model,original_layer_counter,obf_model,obf_layer_counter,obf_ratio)
            else:
                original_layer_counter,obf_layer_counter = resnet_convert_conv_block_normal(model,original_layer_counter,obf_model,obf_layer_counter,obf_ratio)
            time2 = time()
            print("stack:{},res_block:{},time:{:.2f}".format(stack,res_block,time2-time1))
    
    ## fully-connected layers
    orig_dense_layer = model.layers[155]
    assert('dense' in orig_dense_layer.name)
    obf_dense_layer = obf_model.layers[288]
    assert('dense' in obf_dense_layer.name)

    obf_dense_layer.set_weights(orig_dense_layer.get_weights())

    # store converted model in file
    obf_model.save(filled_model)

    return



if __name__ == '__main__':
    depth = 44
    orig_model = "resnet-{}.h5".format(depth)
    obf_custom_tpl = 'resnet-{}-trans-empty.h5'.format(depth)
    filled_model= 'resnet-{}-trans-filled.h5'.format(depth)
    obf_ratio = 1.2
    resnet_converter_custom(orig_model, obf_custom_tpl, filled_model, obf_ratio)
