#!/usr/bin/env python
from numpy import asarray
from numpy import save
from numpy import load
import numpy as np
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel 

def get_weights_info(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    
    for name, weight in zip(names, weights):
        print(name, weight.shape)

def print_layer_var(model, idx):
    #idx == 6: add_mask layer
    am = model.layers[idx]
    print(dir(am))
    print(am.random_scalar)

def store_random_scalar(random_scalar):
    rs = np.array([random_scalar], dtype=np.float32)
    rs.tofile(fh)
    print("t.shape:%s, t.type: %s"%(t[0].shape, type(t)))

# idx == 4, batchnorm layer
def store_layer_weights(model, idx):
    bn =  model.layers[idx]
    print(dir(bn))
    t =  model.layers[idx].get_weights()
    w =  model.layers[idx].weights
    print("type w:%s" % type(w))
    fh = open('testweight.bin','b+w')
    t[0].tofile(fh)


# load model
ORIG_MODEL='mobilenet_obf_filled_plan_A.h5'
#ORIG_MODEL='testnet_obf_filled.h5'
model = tf.keras.models.load_model(ORIG_MODEL, custom_objects={'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})

