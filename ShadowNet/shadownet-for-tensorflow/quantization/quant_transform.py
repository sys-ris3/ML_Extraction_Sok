#!/usr/bin/env python
import tensorflow as tf
import pandas as pd
import pickle
import os,sys
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import argparse
import functools
import time
from tensorflow.python.keras.engine import training
from tensorflow.python.keras import layers
from tensorflow.keras import initializers, Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer,Conv2D,Dense,InputLayer,AveragePooling2D,MaxPooling2D,Activation,Flatten,Reshape,Dropout,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,Lambda,DepthwiseConv2D,ReLU, Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.nn import relu6

from tensorflow.keras.utils import plot_model

from IPython  import embed
cur_dir = os.path.dirname(__file__)
resnet_package_path = os.path.join(cur_dir,"..","eval-networks","resnet")
sys.path.append(resnet_package_path)

from resnet import ResNetBlock

from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric 
from linear_transform_layer import LinearTransform
from tee_shadow_generic_layer import TeeShadowGeneric
from shuffle_channel_layer import ShuffleChannel 
from tee_shadow_layer import TeeShadow
from tee_shadow_generic_layer import TeeShadowGeneric
from tee_shadow_generic_2inputs import TeeShadowGeneric2Inputs

import random
import itertools
# part of the code here borrowed from slalom https://github.com/ftramer/slalom/blob/master/python/slalom/quant_layers.py

# To quicken the conversion process, we create a key-value weights cache
# key: hash of weights before convertion
# value: converted weights
WEIGHTS_CACHE_PATH = "weights_cache.pkl"
WEIGHTS_CACHE = {}
# set seed to get fixed hash value for same input
PYTHONHASHSEED=0

USE_CACHE = False 
LITTLE_WEIGHTS = False

P = 2**23 + 2**21 + 7
INV_P = 1.0 / P
MID = P // 2
assert(P + MID < 2**24)
q = float(round(np.sqrt(MID))) + 1
inv_q = 1.0 / q

def current_milli_time():
    return round(time.time() * 1000)

def get_all_layers(model):
    all_layers = [[l] for l in model.layers]
    all_layers = list(itertools.chain.from_iterable(all_layers))
    return all_layers

def get_first_linear_layer_idx(layers):
    for i in range(len(layers)):
        if isinstance(layers[i], Conv2D) or isinstance(layers[i],Dense):
            return i
    return -1

def get_last_linear_layer_idx(layers):
    num = len(layers)
    for i in range(num-1, -1, -1):
        if isinstance(layers[i], Conv2D) or isinstance(layers[i],Dense):
            return i
    return -1

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
    obf_chn = int(np.ceil(chn * obf_ratio))
    obf_shape = shape[:-1] + (obf_chn,)  

    # generate ndarray to store obfuscated weight
    obf_weight = np.ndarray(shape=obf_shape, dtype=float, order='F')

    # generate random part of weight for obfuscation
    rand_weight_shape = shape[:-1] + ((obf_chn - chn),)
    rand_weight = tf.random.normal(rand_weight_shape)
    if (LITTLE_WEIGHTS): # original model weights might be small, use little weights
        rand_weight = rand_weight/50.

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
        #scalar = random.random()
        # no need to apply scalar when input and output are masked already
        scalar = 1.0
        obf_dict[i] = (main_chn, rand_chn, scalar)

    # obfuscation
    start = time.time()
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
    end = time.time()
    print(end - start) 

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

"""
Assume channel last input format
@params:
    @dwconv_weight:
@return:
    @obf_dwconv_weight
    @dwconv_shuffle_weight
    @dwconv_restore_weight
"""
def generate_dwconv_obf_weight(dwconv_ws):
    
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
    assert(len(dwconv_ws) == 2) # weights + bias after fuse_bn
    dwconv_w = dwconv_ws[0]
    dwconv_b = dwconv_ws[1]
    chn = dwconv_w.shape[-2]
    arr = np.arange(chn)
    np.random.shuffle(arr)

    #dictionary used for 
    obf_dict = {}
    for i in range(chn):
        idx_to = np.where(arr == i)[0][0]
        idx_from = arr[i]
        #scalar = random.random()
        scalar = 1.0
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
    obf_dwconv_b = np.ndarray(shape=dwconv_b.shape, dtype=float, order='F')
    for c in range(dwconv_w.shape[2]):
        idx_from = obf_dict[c][1]
        scalar = obf_dict[idx_from][2]
        for i in range(dwconv_w.shape[0]):
            for j in range(dwconv_w.shape[1]):
                for k in range(dwconv_w.shape[3]):
                    obf_dwconv_w[i][j][c][k] = dwconv_w[i][j][idx_from][k] * scalar
        obf_dwconv_b[c] = dwconv_b[idx_from] * scalar
    obf_dwconv_ws = [obf_dwconv_w, obf_dwconv_b]
    return obf_dwconv_ws, shuffle_weights, restore_weights 

def convert_depthconv_weights(old_depth, new_depth, shuffle_channel_inp, shuffle_channel_outp, quantize, bits_w, bits_x):
    dwconv_w = old_depth.get_weights()
    obf_dwconv_ws, shuffle_weights, restore_weights = generate_dwconv_obf_weight(dwconv_w)
    obf_dwconv_w = obf_dwconv_ws[0]
    obf_dwconv_b = obf_dwconv_ws[1]

    if quantize:
        range_w = 2**bits_w
        range_x = 2**bits_x
        obf_dwconv_wq = np.round(range_w * obf_dwconv_w)
        obf_dwconv_bq = np.round(range_x * range_w * obf_dwconv_b)
        new_depth.set_weights([obf_dwconv_wq, obf_dwconv_bq])
    else:
        new_depth.set_weights([obf_dwconv_w, obf_dwconv_b])

    # shuffle channel not affected by quantize
    shuffle_channel_inp.set_weights(shuffle_weights)
    shuffle_channel_outp.set_weights(restore_weights)
    return

def quantize_dense_weights(dense_layer, new_conv, quantize, bits_w, bits_x):
    conv_weights = np.reshape(dense_layer.get_weights()[0], (1,1)+dense_layer.get_weights()[0].shape)
    if quantize:
        range_w = 2**bits_w
        range_x = 2**bits_x
        obf_weight_q = np.round(range_w * conv_weights)
        bias_q = np.round(range_x * range_w * dense_layer.get_weights()[1])
        new_conv.set_weights([obf_weight_q, bias_q])

    else:
        bias = dense_layer.get_weights()[1]
        new_conv.set_weights([conv_weights, bias])
    return
    
def subsample_hash(a, obf_ratio):
    rng = np.random.RandomState(89)
    inds = rng.randint(low=0, high=a.size, size=10)
    b = a.flat[inds] 
    b.flags.writeable = False
    print(b[:5])
    return hash(bytes(b.data))

def convert_conv_weights(obf_ratio, old_conv, new_conv, linear_transform, is_dense,useBias, quantize, bits_w, bits_x,cache=None):
    global WEIGHTS_CACHE
    global USE_CACHE
    if cache != None:
        WEIGHTS_CACHE = cache
        USE_CACHE = True
    print("old:%s, new:%s, lt:%s, isDense:%s"%(old_conv.name, new_conv.name, linear_transform.name, str(is_dense)))
    # obf_conv weights transformation 
    if is_dense:
        conv_weights = np.reshape(old_conv.get_weights()[0], (1,1)+old_conv.get_weights()[0].shape)
    else:
        conv_weights = old_conv.get_weights()[0]

    if USE_CACHE:
        key = subsample_hash(conv_weights, obf_ratio)
        print("key:" + str(key))
        if key not in WEIGHTS_CACHE:
            print("generating cache")
            arr, obf_dict, obf_weight = normal_conv_kernel_obf(conv_weights, obf_ratio)
            WEIGHTS_CACHE[key] = (arr, obf_dict, obf_weight) 
        else:
            print("re-using cache")
            arr, obf_dict, obf_weight = WEIGHTS_CACHE[key]
    else:
        print("did not use cache")
        arr, obf_dict, obf_weight = normal_conv_kernel_obf(conv_weights, obf_ratio)
        

    # populate linearTransform layer's weights
    m_weight = linear_transform.get_weights()[0]
    s_weight = linear_transform.get_weights()[1]
    populate_linear_transform_weight(m_weight, s_weight, arr, obf_dict)

    if quantize:
        range_w = 2**bits_w
        range_x = 2**bits_x
        obf_weight_q = np.round(range_w * obf_weight)
        if useBias:
            bias_q = np.round(range_x * range_w * old_conv.get_weights()[1])
        else:
            bias_q = np.zeros(s_weight.shape)
            print("bias_q shape")
            print(bias_q.shape)
        new_conv.set_weights([obf_weight_q])

        # m_weight store index
        # s_weight store scalar, default to 1.0
        linear_transform.set_weights([m_weight,s_weight, bias_q])
    else:
        bias = old_conv.get_weights()[1]
        new_conv.set_weights([obf_weight])
        linear_transform.set_weights([m_weight,s_weight, bias])
    return

# new_layers start with push_mask, end with pop_mask, usually with conv+lt in between
def convert_mask_weights(new_layers, old_layer, quantize, bits_w, bits_x):
    push_mask = new_layers[0]
    pop_mask = new_layers[-1]
    assert(len(new_layers) > 1)
    assert(isinstance(push_mask,AddMask))
    assert(isinstance(pop_mask,AddMask))
    #assert(isinstance(old_layer,Conv2D))
    print("mask old_layer:",old_layer)
    input_mask_weights = push_mask.get_weights()

    # for evaluation of mask generation time
    rand_weight = tf.random.normal(input_mask_weights[0].shape)
    zero_weight = tf.zeros(input_mask_weights[0].shape)


    if quantize:
        range_x = 2**bits_x
        range_w = 2**bits_w
        input_mask_weights_q = [np.round(input_mask_weights[0] * range_x)]
        push_mask.set_weights(input_mask_weights_q)

        xq = tf.convert_to_tensor(input_mask_weights_q)

        xq = old_layer(xq)

        # get old_layer bias with zero input, remove its impact on output
        xq_0 = old_layer(np.asarray([zero_weight]))
        xq -= xq_0

        xq_np = xq.numpy()
        xq_np = xq_np*range_w # original layer weights not quantized, add it
        xq_np[xq_np >= MID] -= P
        xq_np[xq_np < -MID] += P
        xq = tf.convert_to_tensor(xq_np)
        output_mask_weight_q = np.array(xq)
        pop_mask.set_weights(output_mask_weight_q)
    else:
        # populate pop-mask layer weights
        x = tf.convert_to_tensor(input_mask_weights)

        ## 
        x_0 = old_layer(np.asarray([zero_weight]))
        x = old_layer(x)
        output_mask_weight = np.array(x - x_0)
        pop_mask.set_weights(output_mask_weight)
    return

def check_model_weights(new_model):
    for l in new_model.layers:
        if isinstance(l, Conv2D):
            print (l.get_weights())

# quantized layer for any non-linear computation
class ActivationQ(Layer):

    def __init__(self, activation, bits_w, bits_x, quantize=True,
                 **kwargs):
        super(ActivationQ, self).__init__(**kwargs)
        self.bits_w = bits_w 
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.quantize = quantize
        self.activation = activation
        assert activation in ["relu", "relu6", "softmax"]

    def activation_name(self):
        return self.activation

    def call(self, inputs):
        #inputs = tf.Print(inputs, [tf.reduce_sum(tf.abs(tf.cast(inputs, tf.float64)))], message="relu input: ")
        #inputs = tf.Print(inputs, [], message="in ActivationQ with input shape: {}".format(inputs.get_shape().as_list()))

        if self.quantize:
            inputs_dq = inputs/(self.range_x * self.range_w)
            if self.activation in ["relu", "relu6"]:
                if self.activation.endswith("relu6"):
                    act = relu(inputs, max_value=6 * self.range_x * self.range_w)
                else:
                    act = relu(inputs)

                outputs = tf.math.round(act / self.range_w)
            else: # softmax
                outputs = tf.nn.softmax(inputs_dq)
        else:
            if self.activation == "relu":
                outputs = tf.nn.relu(inputs)
            elif self.activation == "relu6":
                outputs = tf.nn.relu6(inputs)
            else:
                outputs = tf.nn.softmax(inputs)
        return outputs 

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'activation': self.activation,
        }
        base_config = super(ActivationQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

# fuse batchnorm layers
def fuse_bn(layers):
    def fuse_bn_op(layer):
        print("called fuse")
        assert(isinstance(layer,BatchNormalization))
        input = layer.get_input_at(0)
        prev_layer = [l for l in layers if (l.get_output_at(0) is input)]
        assert len(prev_layer) == 1
        conv = prev_layer[0]

        assert isinstance(conv, Conv2D) or isinstance(conv, DepthwiseConv2D)
        assert layer.axis[0] == 3 or layer.axis[0] == -1

        mean = layer.moving_mean
        var = layer.moving_variance
        beta = layer.beta if layer.beta is not None else 0.0
        gamma = layer.gamma if layer.gamma is not None else 1.0

        w = conv.get_weights()[0]
        b = 0

        # conv layer had no bias
        if not conv.use_bias:
            if isinstance(conv, DepthwiseConv2D):
                bias_shape = (conv.depthwise_kernel.get_shape().as_list()[2],)
            else:
                bias_shape = (conv.filters,)

            conv.bias = conv.add_weight(shape=bias_shape,
                            initializer=initializers.get('zeros'),
                            name='bias',
                            regularizer=None,
                            constraint=None)

            conv.use_bias = True
        b = conv.get_weights()[1]

        if isinstance(conv, DepthwiseConv2D):
            w = np.transpose(w, (0, 1, 3, 2))

        new_w = w * gamma / np.sqrt(var + layer.epsilon)
        new_b = (b-mean) * gamma / (np.sqrt(var + layer.epsilon)) + beta

        if isinstance(conv, DepthwiseConv2D):
            new_w = np.transpose(new_w, (0, 1, 3, 2))

        conv.set_weights([new_w, new_b]) 
    
    for (i, layer) in enumerate(layers):
        if isinstance(layer, BatchNormalization):
            fuse_bn_op(layer)
        if isinstance(layer, ResNetBlock):
            fuse_bn(layer.layer1_functors)
            fuse_bn(layer.layer2_functors)
            if hasattr(layer,"layer3_functors"):
                fuse_bn(layer.layer3_functors)
            fuse_bn(layer.layer4_functors)
    return 


"""
Compare tensor a and b of shape(B, H, W, C) elementwisely 
and return the accumulated errors.
"""
def compare_tensor(a, b):
    # store accumulated errors on all tensor elements
    abs_errs = 0
    err_ratios = []
    abs_vals = 0 

    if a.shape != b.shape:
        print ("two tensors are in different shape")
        print ("a.shape:" + a.shape)
        print ("b.shape:" + b.shape)
        return

    if len(a.shape) == 2:
        a = tf.convert_to_tensor(np.reshape(a, (1,1)+a.shape), dtype=tf.float32)
        b = tf.convert_to_tensor(np.reshape(b, (1,1)+b.shape), dtype=tf.float32)

    # sampling tensor values to compare
    for cnt in range(10):
        i = 0 
        j =  random.randint(0, a.shape[1] - 1)
        k =  random.randint(0, a.shape[2] - 1)
        c =  random.randint(0, a.shape[3] - 1)
        err = abs(a[i][j][k][c] - b[i][j][k][c])
        if err > 0.00001: # report on big error
            print ("i:%d, j:%d, k:%d, err:%f"%(i,j,k,err))

        abs_errs += abs(err)

        error_by_a = abs(err/a[i][j][k][c])
        error_by_b = abs(err/b[i][j][k][c])

        # original value of a/model
        abs_vals += abs(a[i][j][k][c].numpy())

        # report middle result because it may take long to finish
        err_ratios.append((error_by_a.numpy(), error_by_b.numpy()))

    avg_vals = abs_vals/10.0
    avg_errs = abs_errs/10.0
    avg_err_ratios = list(map(lambda x: x/len(err_ratios), functools.reduce(lambda x, y: (x[0]+y[0],x[1]+y[1]), err_ratios))) 
    print ("errs:%f, avg_err:%f"%(abs_errs, abs_errs/10.0))
    print ("err ratios:")
    print (err_ratios)
    print ("original model output:")
    print (avg_vals)
    return avg_errs, avg_err_ratios, avg_vals

def run_submodel(submodel, inp):
    x = inp
    for l in submodel:
        x = l(x)
    return x
    
def compare_block(quantize, orig_block, trans_block):
    print("compare block:")
    print("orig block")
    for x in orig_block:
        print(x.name)
    print("trans block")
    for x in trans_block:
        print(x.name)
    print("")

    print(orig_block[0].input_shape)
    print(trans_block[0].input_shape)
    input_shape = orig_block[0].input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    layer_input = tf.random.uniform((1,) + input_shape[1:])

    if quantize:
        trans_input = layer_input*256.0
    else:
        trans_input = layer_input
        
    out2 = run_submodel(trans_block, trans_input)
    out1 = run_submodel(orig_block, layer_input)

    last_layer = orig_block[-1]
    conf = last_layer.get_config()
    if quantize:
        if (isinstance(last_layer, Dense)) and (conf['activation'] == 'softmax'):
            out2 = out2  
        else:
            out2 = out2/256.0/256.0  

    avg_errs, avg_err_ratios, avg_vals = compare_tensor(out1, out2)
    return avg_errs, avg_err_ratios, avg_vals 

def evaluate_transformed_model(quantize, orig_model, trans_model, orig_blocks, conv_trans_blocks, mask_trans_blocks, orig_chkpt, trans_chkpt, fig_name): 

    # error compared with original model's output 
    block_errs_no_mask = [] # trans w/o mask 
    block_errs_with_mask = [] # trans w/ mask 
    trans_model_chkpt_errs = [] # trans

    block_err_ratios_no_mask = [] # trans w/o mask
    block_err_ratios_with_mask = [] # trans w/ mask
    trans_model_chkpt_err_ratios = [] # trans

    block_vals_no_mask = [] # trans w/o mask
    block_vals_with_mask = [] # trans w/ mask
    trans_model_chkpt_vals = [] # trans

    for i in range(len(orig_blocks) - 1):
        avg_errs, avg_err_ratios, avg_vals = compare_block(quantize, orig_blocks[i], conv_trans_blocks[i])
        block_errs_no_mask.append(avg_errs)
        block_err_ratios_no_mask.append(avg_err_ratios)
        block_vals_no_mask.append(avg_vals)

        avg_errs, avg_err_ratios, avg_vals = compare_block(quantize, orig_blocks[i], mask_trans_blocks[i])
        block_errs_with_mask.append(avg_errs)
        block_err_ratios_with_mask.append(avg_err_ratios)
        block_vals_with_mask.append(avg_vals)

        avg_errs, avg_err_ratios, avg_vals = compare_block(quantize, orig_model.layers[:orig_chkpt[i]], trans_model.layers[:trans_chkpt[i]])
        trans_model_chkpt_errs.append(avg_errs)
        trans_model_chkpt_err_ratios.append(avg_err_ratios)
        trans_model_chkpt_vals.append(avg_vals)

    """
    b:block, e:errs, n:no mask, m: mask/model, r:error ratio
    """
    ben = block_errs_no_mask
    bem = block_errs_with_mask
    me  = trans_model_chkpt_errs

    brn = block_err_ratios_no_mask
    brm = block_err_ratios_with_mask
    mr = trans_model_chkpt_err_ratios

    bvn = block_vals_no_mask
    bvm = block_vals_with_mask
    mv = trans_model_chkpt_vals

    plt.xlabel('layer id')
    plt.ylabel('error ratio(er)/output(val)')
    plt.plot([e[0] for e in brn], 'gs', lw=2, label='layer nm(er)') 
    plt.plot([e[0] for e in brm], 'y+', lw=2, label='layer wm(er)') 
    plt.plot([e[0] for e in mr], 'r+', lw=2, label='model (er)')
    plt.plot(bvn, 'bo', lw=2, label='layer nm(val)')
    plt.plot(bvm,'go', lw=2, label='layer wm(val)')
    plt.plot(mv,'rs', lw=2, label='model (val)')

    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .402), loc='lower left',
                       ncol=3, mode="expand", borderaxespad=0.)
    print("save " + fig_name)
    plt.savefig(fig_name)

def get_act_quant(layer, bits_w, bits_x, quantize, layer_idx):
    act_func = "relu6" if layer.activation == relu6 else "relu" if layer.activation == relu else "softmax"
    actq = ActivationQ(act_func, bits_w, bits_x, quantize, name = "activation_%d"%layer_idx)
    return actq

def build_functional_model_from_layers(new_model_layers, model_name):
    print(new_model_layers[0].input_shape[0][1:])
    model_input = layers.Input(shape=new_model_layers[0].input_shape[0][1:])
    assert(isinstance(new_model_layers[0], InputLayer))
    l = new_model_layers[1]
    print(model_input.shape)
    x = l(model_input)
    for l in new_model_layers[2:]:
        x = l(x)
    model = training.Model(model_input, x, name=model_name) 
    model.build(model.layers[0].input_shape)
    return model

conv_map = {} 
depth_map = {} 
dense_map = {} 
mask_box = []
conv_id = 0

def transform_layer(model_tag, layer, layer_idx, is_first_linear, is_last_linear, obf_ratio = 1.2, quantize=True, keep_dense=True, bits_w = 8, bits_x = 8):
    print("transform {})".format(layer))
    new_layers = []
    orig_block = []
    conv_trans_block = []
    mask_trans_block = []

    if isinstance(layer, InputLayer):
        new_layers.append(InputLayer.from_config(layer.get_config()))

    elif isinstance(layer, DepthwiseConv2D):
        conf = layer.get_config()
        
        assert conf['activation'] == "linear"
        
        push_mask = AddMask(random_scalar=1.0)
        new_layers.append(push_mask)

        shuffle_inp = ShuffleChannel(name="shuffle_channel_in_%d" % layer_idx)
        new_layers.append(shuffle_inp)

        # keep DepthwiseConv2D conf for quantization
        new_depth = DepthwiseConv2D.from_config(conf)
        new_layers.append(new_depth)

        shuffle_outp = ShuffleChannel(name="shuffle_channel_out_%d" % layer_idx)
        new_layers.append(shuffle_outp)

        pop_mask = AddMask(random_scalar= -1.0)
        new_layers.append(pop_mask)

        # weights transformation
        depth_map[layer] = (shuffle_inp, new_depth, shuffle_outp) 
        mask_box.append((new_layers, len(new_layers), layer)) 

        # prepare blocks for transformation evaluation
        orig_block.append(layer)
        conv_trans_block.extend(new_layers[1:-1]) # exclude masks
        mask_trans_block.extend(new_layers)

    elif isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
        conf = layer.get_config()
        useBias = False
        isDense = False

        # push mask if not first linear layer 
        if not is_first_linear:
            push_mask = AddMask(random_scalar=1.0)
            new_layers.append(push_mask)

        act = conf['activation']
        conf['activation'] = "linear"
        global conv_id
        conf['name'] = model_tag + "_conv2d_{}".format(conv_id)
        conv_id += 1
        filters = conf['filters']
        obf_filters = int(np.ceil(filters*obf_ratio))
        conf['filters'] = obf_filters

        conf['use_bias'] = False
        # bias is added at LinearTransformGeneric layer
        if layer.use_bias:
            useBias = True

        new_conv = Conv2D.from_config(conf)
        new_layers.append(new_conv)

        lt = LinearTransformGeneric(filters)
        new_layers.append(lt)

        # weights transformation
        conv_map[layer] = (new_conv, lt, isDense, useBias) 

        if not is_first_linear: # pop mask (salar=-1) if not first linear layer 
            pop_mask = AddMask(random_scalar= -1.0)
            new_layers.append(pop_mask)
            mask_box.append((new_layers, len(new_layers), layer)) 

        if act != 'linear':
            act_layer = ActivationQ(act, bits_w, bits_x, quantize, name='activation_%d'%layer_idx)
            new_layers.append(act_layer)

        # prepare conv_trans and mask_trans for block comparison
        orig_block.append(layer) 

        conv_trans_block.append(new_conv)
        conv_trans_block.append(lt)
        if act != 'linear':
            conv_trans_block.append(act_layer) 

        mask_trans_block.extend(new_layers)

    elif isinstance(layer, Dense):
        conf = layer.get_config()
        useBias = False
        isDense = True 

        conf['use_bias'] = False
        if layer.use_bias and not keep_dense:
            useBias = True 

        filters = layer.units
        obf_filters = int(np.ceil(filters*obf_ratio))
        if keep_dense:
            conf['filters'] = filters 
        else:
            conf['filters'] = obf_filters 
        del conf['units']

        conf['kernel_size'] = 1

        act = conf['activation']
        conf['activation'] = 'linear' 

        if not is_first_linear: # push mask if not first linear layer 
            push_mask = AddMask(random_scalar=1.0)
            new_layers.append(push_mask)

        h_in = int(layer.input_spec.axes[-1])
        inp_reshape = Reshape((1, 1, h_in))
        new_layers.append(inp_reshape)


        new_conv = Conv2D.from_config(conf)
        new_layers.append(new_conv)

        if keep_dense:
            dense_map[layer] = new_conv 
        else:
            lt = LinearTransformGeneric(filters)
            new_layers.append(lt)

            # transform conv weights
            conv_map[layer] = (new_conv, lt, isDense, useBias)


        reshape = Reshape((filters,))
        new_layers.append(reshape)

        if not is_first_linear: # pop mask if not first linear layer 
            pop_mask = AddMask(random_scalar=-1.0)
            new_layers.append(pop_mask)
            # convert_mask_weights(new_layers)
            mask_box.append((new_layers,len(new_layers), layer)) # skip reshape

        if act != 'linear':
            act_layer = ActivationQ(act, bits_w, bits_x, quantize, name='activation_%d'%layer_idx)
            new_layers.append(act_layer)
        else:
            act_layer = None


        orig_block.append(layer)
        if act_layer != None:
            if keep_dense:
                conv_trans_block.extend([inp_reshape, new_conv, reshape, act_layer])
            else:
                conv_trans_block.extend([inp_reshape, new_conv, lt, reshape, act_layer])
        else:
            if keep_dense:
                conv_trans_block.extend([inp_reshape, new_conv, reshape])
            else:
                conv_trans_block.extend([inp_reshape, new_conv, lt, reshape])
        mask_trans_block.extend(new_layers)

    elif isinstance(layer, BatchNormalization):
        pass

    elif isinstance(layer, MaxPooling2D):
        new_layers.append(MaxPooling2D.from_config(layer.get_config()))

    elif isinstance(layer, AveragePooling2D):
        new_layers.append(AveragePooling2D.from_config(layer.get_config()))
        #new_layers.append(Lambda(tf.round))

    elif isinstance(layer, ReLU):
        # MobileNet use ReLU(6.0)
        conf = layer.get_config()
        assert conf['max_value'] == 6.0

        act_func = "relu6"
        new_layers.append(ActivationQ(act_func, bits_w, bits_x, quantize, name = "activation_%d"%layer_idx))
    elif isinstance(layer, Activation):
        print(layer.activation)
        assert layer.activation in [relu, relu6, softmax]
        act_func = "relu6" if layer.activation == relu6 else "relu" if layer.activation == relu else "softmax"
        new_layers.append(ActivationQ(act_func, bits_w, bits_x, quantize, name = "activation_%d"%layer_idx))

    elif isinstance(layer, ZeroPadding2D):
        new_layers.append(ZeroPadding2D.from_config(layer.get_config()))

    elif isinstance(layer, Flatten):
        new_layers.append(Flatten.from_config(layer.get_config()))

    elif isinstance(layer, GlobalAveragePooling2D):
        conf = layer.get_config()
        new_layers.append(GlobalAveragePooling2D.from_config(conf))

    elif isinstance(layer, Reshape):
        reshape_layer = Reshape.from_config(layer.get_config())
        reshape_layer._name = "reshape_%d"%layer_idx
        new_layers.append(reshape_layer)

    elif isinstance(layer, Dropout):
        pass
    elif isinstance(layer, ResNetBlock):
        resnet_conf = layer.get_config()
        resnet_conf["transformed"]= True
        resnet_conf['quant'] = quantize
        resnet_conf['layer_idx'] = layer_idx
        resnet_conf['is_first_linear'] = is_first_linear
        resnet_conf['is_last_linear'] = is_last_linear
        #resnet_conf['bits_w'] = bits_w 
        #resnet_conf['bits_x'] = bits_x
        new_resnet_block = ResNetBlock(**resnet_conf)

        new_resnet_block.layer1_functors.clear()
        for each_layer1_functor in layer.layer1_functors:
            transformed_layers, _ , _ , _ = transform_layer(model_tag, each_layer1_functor, layer_idx, is_first_linear, is_last_linear, obf_ratio, quantize, keep_dense, bits_w, bits_x)
            for each_transformed_layer in transformed_layers:
                new_resnet_block.layer1_functors.append(each_transformed_layer)
        ##layer2
        new_resnet_block.layer2_functors.clear()
        for each_layer2_functor in layer.layer2_functors:
            transformed_layers, _ , _ , _ = transform_layer(model_tag, each_layer2_functor, layer_idx, is_first_linear, is_last_linear, obf_ratio, quantize, keep_dense, bits_w, bits_x)
            for each_transformed_layer in transformed_layers:
                new_resnet_block.layer2_functors.append(each_transformed_layer)
        ## layer3
        if new_resnet_block.stack > 0 and new_resnet_block.res_block == 0:
            new_resnet_block.layer3_functors.clear()
            for each_layer3_functor in layer.layer3_functors:
                transformed_layers, _ , _ , _ = transform_layer(model_tag, each_layer3_functor, layer_idx, is_first_linear, is_last_linear, obf_ratio, quantize, keep_dense, bits_w, bits_x)
                for each_transformed_layer in transformed_layers:
                    new_resnet_block.layer3_functors.append(each_transformed_layer)
        ## layer4
        new_resnet_block.layer4_functors.clear()
        for each_layer4_functor in layer.layer4_functors:
            transformed_layers , _ , _ , _ = transform_layer(model_tag, each_layer4_functor, layer_idx, is_first_linear, is_last_linear, obf_ratio, quantize, keep_dense, bits_w, bits_x)
            for each_transformed_layer in transformed_layers:
                new_resnet_block.layer4_functors.append(each_transformed_layer)
        new_layers.append(new_resnet_block)
    elif tf.python.keras.layers.merge.add.__code__.co_code == layer.__code__.co_code:
        new_layers.append(tf.keras.layers.add)
    else:
        print("unsupported layer", layer)
        embed()
        exit(1)

    return new_layers, orig_block, conv_trans_block, mask_trans_block

# transform a model into a (quantized) shadownet model
def transform(model, model_name, model_tag, obf_ratio = 1.2, quantize=True, keep_dense=True, dry_run = False, fig_name="eval.png", bits_w = 8, bits_x = 8):
    layers = model.layers
    
    # a block is several consecutive layers
    orig_chkpt = []
    trans_chkpt = []

    new_model_layers = []
    fuse_bn(layers)## resnet's bn will be fused later
    
    orig_blocks = []
    conv_trans_blocks = []
    mask_trans_blocks = []

    first_linear_idx = get_first_linear_layer_idx(layers)
    last_linear_idx = get_last_linear_layer_idx(layers)

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        new_layers, orig_block, conv_trans_block, mask_trans_block = transform_layer(model_tag, layer, layer_idx, layer_idx == first_linear_idx, layer_idx == last_linear_idx, obf_ratio, quantize, keep_dense, bits_w, bits_x)
        for new_layer in new_layers:
            new_model_layers.append(new_layer)
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            orig_blocks.append(orig_block)
            conv_trans_blocks.append(conv_trans_block)
            mask_trans_blocks.append(mask_trans_block)

            orig_chkpt.append(layer_idx+1)
            trans_chkpt.append(len(new_model_layers))

    # build new_model
    new_model = build_functional_model_from_layers(new_model_layers, model_name) 
    print(new_model.summary())

    # convert weights
    if not dry_run:
        for l in conv_map:
            (new_conv, lt, isDense, useBias) = conv_map[l]
            convert_conv_weights(obf_ratio, l, new_conv, lt, isDense, useBias, quantize, bits_w, bits_x)
        
        for l in dense_map:
            print("dense_map")
            new_conv = dense_map[l]
            quantize_dense_weights(l, new_conv, quantize, bits_w, bits_x)
        
        start_time = current_milli_time()
        for (new_layers, length, old_layer) in mask_box:
            convert_mask_weights(new_layers[:length], old_layer, quantize, bits_w, bits_x)
        end_time = current_milli_time()
        print("100 mask transformation time %6.2f ms" % (end_time - start_time))

        for l in depth_map:
            (shuffle_inp, new_depth, shuffle_outp) = depth_map[l]
            convert_depthconv_weights(l, new_depth, shuffle_inp, shuffle_outp, quantize, bits_w, bits_x)

    # evaluate transformed model before return
    evaluate_transformed_model(quantize, model, new_model, orig_blocks, conv_trans_blocks, mask_trans_blocks, orig_chkpt, trans_chkpt, fig_name)

    return new_model

def get_next_conv(layers, start):
    idx = start
    while idx < len(layers):
        each_layer = layers[idx]
        if isinstance(each_layer, Conv2D) or isinstance(each_layer, Dense):
            return idx, each_layer.input_shape
        elif isinstance(each_layer, ResNetBlock):
            assert(isinstance(each_layer.layer1_functors[1], Conv2D))
            return idx, each_layer.layer1_functors[1].input_shape
        else:
            idx += 1
    return idx, None

def split_resnet_block(resnet_block, conv_id,final_output_shape):
    new_resnet_block = ResNetBlock(**resnet_block.get_config())
    ## deal with functors 1
    new_resnet_block.layer1_functors.clear()
    layer1_conv = resnet_block.layer1_functors[1]
    assert(isinstance(layer1_conv, Conv2D))
    new_resnet_block.layer1_functors.append(layer1_conv)
    layer_tee_output_shape = resnet_block.layer1_functors[-1].output_shape
    new_resnet_block.layer1_functors.append(TeeShadowGeneric(layer_tee_output_shape[1],layer_tee_output_shape[2],layer_tee_output_shape[3], position="conv2d_{}".format(conv_id), name="ts_conv_{}".format(conv_id)))
    conv_id += 1

    ## deal with functors 2
    new_resnet_block.layer2_functors.clear()
    layer2_conv = resnet_block.layer2_functors[1]
    assert(isinstance(layer2_conv, Conv2D))
    new_resnet_block.layer2_functors.append(layer2_conv)
    layer_tee_output_shape = resnet_block.layer2_functors[-1].output_shape
    # new_resnet_block.layer2_functors.append(TeeShadowGeneric(layer_tee_output_shape[1],layer_tee_output_shape[2],layer_tee_output_shape[3], position="conv2d_{}".format(conv_id), name="ts_conv_{}".format(conv_id)))
    conv_id += 1

    ## deal with functors 3 if any
    if hasattr(resnet_block, 'layer3_functors'):
        new_resnet_block.layer3_functors.clear()
        layer3_conv = resnet_block.layer3_functors[1]
        assert(isinstance(layer3_conv, Conv2D))
        
        new_resnet_block.layer3_functors.append(layer3_conv)
        layer_tee_output_shape = resnet_block.layer3_functors[-1].output_shape
        #new_resnet_block.layer3_functors.append(TeeShadowGeneric(layer_tee_output_shape[1], layer_tee_output_shape[2],layer_tee_output_shape[3], position="conv2d_{}".format(conv_id),   name="ts_conv_{}".format(conv_id)))
        conv_id += 1
    ## deal with the add
    new_resnet_block.layer4_functors.append(TeeShadowGeneric2Inputs(final_output_shape[1], final_output_shape[2],final_output_shape[3], position="conv2d_{}".format(conv_id),  name="ts_conv_{}".format(conv_id)))
    conv_id += 1
    return new_resnet_block, conv_id

def split_resnet_block_funtional(model_tag, inputs,resnet_block, conv_id,final_output_shape):
    x = inputs
    y = inputs
    ## deal with functors 1
    layer1_conv = resnet_block.layer1_functors[1]
    assert(isinstance(layer1_conv, Conv2D))
    x = layer1_conv(x)
    layer_tee_output_shape = resnet_block.layer1_functors[-1].output_shape
    x = TeeShadowGeneric(layer_tee_output_shape[1],layer_tee_output_shape[2],layer_tee_output_shape[3], 
    position="{}_conv{}".format(model_tag, conv_id), name="ts_conv_{}".format(conv_id))(x)
    conv_id += 1

    ## deal with functors 2
    layer2_conv = resnet_block.layer2_functors[1]
    assert(isinstance(layer2_conv, Conv2D))
    ##print("name:{}\n".format(layer2_conv.name))
    x = layer2_conv(x)
    layer_tee_output_shape = resnet_block.layer2_functors[-1].output_shape
    #x = TeeShadowGeneric(layer_tee_output_shape[1],layer_tee_output_shape[2],layer_tee_output_shape[3], position="conv2d_{}".format(conv_id), name="ts_conv_{}".format(conv_id))(x)
    conv_id += 1

    ## deal with functors 3 if any
    if hasattr(resnet_block, 'layer3_functors'):
        layer3_conv = resnet_block.layer3_functors[1]
        assert(isinstance(layer3_conv, Conv2D))
        #print("name:{}\n".format(layer3_conv.name))
        y = layer3_conv(y)
        layer_tee_output_shape = resnet_block.layer3_functors[-1].output_shape
        #y = TeeShadowGeneric(layer_tee_output_shape[1], layer_tee_output_shape[2],layer_tee_output_shape[3], position="conv2d_{}".format(conv_id),   name="ts_conv_{}".format(conv_id))(y)
        conv_id += 1
    ## deal with the add
    x = TeeShadowGeneric2Inputs(final_output_shape[1], final_output_shape[2],final_output_shape[3], position="{}_conv{}".format(model_tag,conv_id),  name="tsi_conv_{}".format(conv_id))([x,y])
    #conv_id += 1
    return x, conv_id

def generate_split_model(new_model, split_name, model_tag):

    layers = new_model.layers

    # set input shape
    input = Input(shape = new_model.input_shape[1:])
    x = input
    layer_idx = 0
    conv_count = 0
    while layer_idx < len(layers):
        layer = layers[layer_idx]
        
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            x = layer(x)
            # prepare tee_shadow_generic
            next_conv_id, next_input_shape = get_next_conv(layers, layer_idx+1)
            if next_input_shape != None:
                assert(len(next_input_shape) == 4)
                h = next_input_shape[1]
                w = next_input_shape[2]
                c = next_input_shape[3]
                pos = model_tag + "_conv%d"%conv_count
                tee_shadow = TeeShadowGeneric(h, w, c, position=pos, name = "ts_conv_%d"%conv_count)
            else:
                output_shape = layers[-1].output_shape
                tee_shadow = TeeShadowGeneric(1, 1, output_shape[-1], "results", name = "ts_conv_%d"%conv_count)
            x = tee_shadow(x)
            conv_count += 1
            layer_idx = next_conv_id 
        elif isinstance(layer, ResNetBlock):
            ## get the output shape for this whole resnet block(more specifically, for the TeeGeneric_2Inputs)
            next_conv_id, next_input_shape = get_next_conv(layers, layer_idx+1)
            if(next_input_shape == None):
                next_input_shape = (None, 1, 1, layers[-1].output_shape)
            assert(len(next_input_shape) == 4)
            x, conv_count = split_resnet_block_funtional(model_tag, x ,layer,conv_count, next_input_shape)
            layer_idx = next_conv_id
        else:
            layer_idx += 1

    split_model = Model(inputs=input, outputs=x, name="split_model")
    split_model.save(split_name)
    print("split model")
    print(split_model.summary())
    plot_model(split_model,split_name + '_layout.png')
    return

def get_trans_model_name(args):
    orig_model = os.path.basename(args.modelpath)
    suffix = '_auto_obf'
    if args.quantize_model:
        suffix += '_quant'
    if args.keep_dense:
        suffix += '_dense'
    suffix += '.h5'
    trans_model = orig_model.split('.')[0]+suffix
    return trans_model

def get_split_model_name(args):
    orig_model = os.path.basename(args.modelpath)
    split_model = orig_model.split('.')[0]+'_split.h5'
    return split_model

"""
Note:if experimental_new_converter set False, it will use TOCO, 
which generate tflite model without output shape!
We should set it to True(default), so MLIR will be used!
"""
def convert_to_tflite(model_name):
    print("convert to tflite")
    model = tf.keras.models.load_model(model_name, custom_objects={'LinearTransform':LinearTransform,'ActivationQ':ActivationQ,'TeeShadow':TeeShadow,'TeeShadowGeneric':TeeShadowGeneric,'AddMask':AddMask,'ShuffleChannel':ShuffleChannel,'LinearTransformGeneric':LinearTransformGeneric,'TeeShadowGeneric2Inputs':TeeShadowGeneric2Inputs,'ResNetBlock':ResNetBlock })
    model.summary()
    model.build(model.layers[0].input_shape)
    model.compile(optimizer='sgd', loss='mean_squared_error')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.experimental_new_converter = False 
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    tflite_model_name = model_name.split('.')[0]+'.tflite'
    open(tflite_model_name, "wb").write(tflite_model)
    return

def ensure_pythonhashseed(seed=0):
    current_seed = os.environ.get("PYTHONHASHSEED")

    seed = str(seed)
    if current_seed is None or current_seed != seed:
        print(f'Setting PYTHONHASHSEED="{seed}"')
        os.environ["PYTHONHASHSEED"] = seed
        # restart the current process
        os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == '__main__':
    '''
    to transform and quantize a model just pass `-q -f`
    '''
    parser = argparse.ArgumentParser(prog='quant_transform')
    parser.add_argument('modelpath',
            help = 'path to model file')
    parser.add_argument('-m', '--model-tag', help='model tag to identity model type:mobilenet|minivgg|resnet|alexnet')
    parser.add_argument('-q', '--quantize-model', action='store_true',
            help = 'quantize the model while performing ShadowNet transform')
    parser.add_argument('-k', '--keep-dense', action='store_true',
            help = 'keep dense layer unchanged while performing ShadowNet transform')
    parser.add_argument('-d', '--dry-run', action='store_true',
            help = 'skip slow weights transformation for debugging')
    parser.add_argument('-c', '--convert-only', action='store_true',
            help = 'only convert given split model to tflite model')
    parser.add_argument('-f', '--fast-run', action='store_true',
            help = 'fast run with weights cache')

    args = parser.parse_args()

    if args.convert_only:
        convert_to_tflite(args.modelpath)
        exit(0)
        
    # load weights cache
    if exists(WEIGHTS_CACHE_PATH):
        WEIGHTS_CACHE = pd.read_pickle(WEIGHTS_CACHE_PATH) # {key:value}
    else:
        WEIGHTS_CACHE = {}

    ensure_pythonhashseed(seed=0)
    
    model_tag = args.model_tag
    assert(model_tag != None)
    print("model tag:{}".format(model_tag))

    if model_tag == "alexnet":
        LITTLE_WEIGHTS = True

    if args.fast_run:
        USE_CACHE = True

    trans_model_name = get_trans_model_name(args)

    orig_model = args.modelpath
    model = tf.keras.models.load_model(orig_model, custom_objects={'ResNetBlock':ResNetBlock})
    plot_model(model, orig_model + '_layout.png',expand_nested=False,dpi=96)
    start_time = time.time()
    new_model = transform(model, trans_model_name, model_tag, 1.2, args.quantize_model, args.keep_dense, args.dry_run, trans_model_name+'.png')
    with open("resnet-404-time.txt","w") as wfile:
        wfile.write("--- %s seconds ---" % (time.time() - start_time))
    # save weights_cache
    pd.to_pickle(WEIGHTS_CACHE,WEIGHTS_CACHE_PATH)
    #embed()
    new_model.save(trans_model_name)
    plot_model(new_model,trans_model_name+'_layout.png',expand_nested=False,dpi=96)
    convert_to_tflite(trans_model_name)
    split_name = get_split_model_name(args)
    generate_split_model(new_model, split_name, model_tag)
    convert_to_tflite(split_name)

