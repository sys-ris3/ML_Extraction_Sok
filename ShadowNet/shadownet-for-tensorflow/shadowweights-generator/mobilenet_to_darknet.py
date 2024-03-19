#!/usr/bin/env python
from numpy import asarray
from numpy import save
from numpy import load
import numpy as np
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel 

def store_LT(fh, mdl, idx):
    # store LT weights
    # m, (2, units)
    t = mdl.layers[idx].get_weights()[0]
    t.tofile(fh)
    # s, (units,)
    t = mdl.layers[idx].get_weights()[1]
    t.tofile(fh)
    return

def store_SF(fh, mdl, idx):
    # store SC shuffle_channel weights
    # m, (1, units)
    t = mdl.layers[idx].get_weights()[0]
    t.tofile(fh)
    # s, (units,)
    t = mdl.layers[idx].get_weights()[1]
    t.tofile(fh)

def store_BN(fh, mdl, idx):
    # store BN BatchNorm weights
    # gamma or scales
    t = mdl.layers[idx].get_weights()[0]
    t.tofile(fh)
    # beta or biases 
    t = mdl.layers[idx].get_weights()[1]
    t.tofile(fh)
    # mean 
    t = mdl.layers[idx].get_weights()[2]
    t.tofile(fh)
    # variance 
    t = mdl.layers[idx].get_weights()[3]
    t.tofile(fh)
    return

def store_AM(fh, mdl, idx):
    #store AM weigths AddMask 
    # mask weight
    t = mdl.layers[idx].get_weights()[0]
    t.tofile(fh)
    # rscalar
    rs = mdl.layers[idx].random_scalar
    rs = np.array([rs], dtype=np.float32)
    rs.tofile(fh)
    return

def store_HD(fh):
    # model header
    major = 0
    minor = 2
    revision = 0
    seen = 0
    nda = np.array([major, minor, revision, seen, seen], dtype=np.int32)
    nda.tofile(fh)
    return

"""
m_type
    A: LT, BN, ReLU, AM, SF
    B: SF, AM, BN, ReLU, AM 
    C: LT, AM, BN, ReLU, AM, SF
    P: LT, AM, BN, ReLU, PL, RS, DO, AM
    R: LT, AM, AM, RS, SM

Layer Abbr:
    LT: LinearTransform
    BN: BatchNorm
    AM: AddMask
    SF: ShuffleChannel
    PL: Avg Pool
    RS: Reshape
    DO: Dropout
    SM: Softmax 

Type Abbr:
    P: Prediction
    R: Results
"""
def generate_darknet_model(name, idx, mdl, m_type):
    fh = open(name, 'b+w')

    # store model header
    store_HD(fh)

    if m_type == "A":
        store_LT(fh, mdl, idx)
        store_BN(fh, mdl, idx + 1)
        store_AM(fh, mdl, idx + 3)
        store_SF(fh, mdl, idx + 4)
    elif m_type == "B":
        store_SF(fh, mdl, idx)
        store_AM(fh, mdl, idx + 1)
        store_BN(fh, mdl, idx + 2)
        store_AM(fh, mdl, idx + 4)
    elif m_type == "C":
        store_LT(fh, mdl, idx)
        store_AM(fh, mdl, idx + 1)
        store_BN(fh, mdl, idx + 2)
        store_AM(fh, mdl, idx + 4)
        store_SF(fh, mdl, idx + 5)
    elif m_type == "P":
        store_LT(fh, mdl, idx)
        store_AM(fh, mdl, idx + 1)
        store_BN(fh, mdl, idx + 2)
        store_AM(fh, mdl, idx + 7)
    elif m_type == "R":
        store_LT(fh, mdl, idx)
        store_AM(fh, mdl, idx + 1)
        store_AM(fh, mdl, idx + 2)
    else:
        print("unrecognized model type! %s" % m_type)
    fh.close()
    return

def convert_mobilenet_to_darknet_models(model, path):
    model_config = {"conv1":("A",3),\
                    "dwconv1":("B",9),\
                    "pwconv1":("C",15),\
                    "dwconv2":("B",23),\
                    "pwconv2":("C",29),\
                    "dwconv3":("B",36),\
                    "pwconv3":("C",42),\
                    "dwconv4":("B",50),\
                    "pwconv4":("C",56),\
                    "dwconv5":("B",63),\
                    "pwconv5":("C",69),\
                    "dwconv6":("B",77),\
                    "pwconv6":("C",83),\
                    "dwconv7":("B",90),\
                    "pwconv7":("C",96),\
                    "dwconv8":("B",103),\
                    "pwconv8":("C",109),\
                    "dwconv9":("B",116),\
                    "pwconv9":("C",122),\
                    "dwconv10":("B",129),\
                    "pwconv10":("C",135),\
                    "dwconv11":("B",142),\
                    "pwconv11":("C",148),\
                    "dwconv12":("B",156),\
                    "pwconv12":("C",162),\
                    "dwconv13":("B",169),\
                    "pwconv13":("P",175),\
                    "results":("R",184)}
    for m in model_config:
        print("%s %s %s"%(m, model_config[m][0], model_config[m][1]))
        generate_darknet_model(path +'/' + m +'.weights', model_config[m][1], model, model_config[m][0])
    return

if __name__ == '__main__':
    #ORIG_MODEL='mobilenet_obf_custom.h5'
    #ORIG_MODEL='mobilenet_obf_filled_plan_A.h5'
    ORIG_MODEL='mobilenet_obf_custom_filled.h5'
    model = tf.keras.models.load_model(ORIG_MODEL, custom_objects={'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    convert_mobilenet_to_darknet_models(model, 'mobilenet-submodels')
