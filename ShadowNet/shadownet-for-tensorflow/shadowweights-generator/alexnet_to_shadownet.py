#!/usr/bin/env python
from numpy import asarray
from numpy import save
from numpy import load
import numpy as np
import tensorflow as tf
import argparse
import os
from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric 

def store_LT(fh, mdl, idx):
    # store LT weights
    # m, (2, units)
    t = mdl.layers[idx].get_weights()[0]
    t.tofile(fh)
    # s, (units,)
    t = mdl.layers[idx].get_weights()[1]
    t.tofile(fh)
    # b, (units,)
    t = mdl.layers[idx].get_weights()[2]
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
model_type
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

def convert_alexnet_to_shadownets(model, outputdir):
    fh = ''
    for i in range(len(model.layers)):
        name = model.layers[i].name
        print("name:" + name)
        if name[:4] == 'conv':
            if name[:6] != 'conv_1':
                fh.close()
            fn = name[:4] + name[5]
            if fn == 'conv9':
                fn = 'results'
            fn += '.weights'
            print("file name:" + fn)
            fh = open(outputdir + '/' + fn, 'w+b')
            store_HD(fh)
        elif name[:4] == 'line':# linear_transform
            store_LT(fh, model, i)
        elif name[:4] == 'push' or name[:4] == 'pop_':
            store_AM(fh, model, i)
        elif name[:4] == 'soft':
            fh.close()
        else:
            continue
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='alexnet_to_shadownet')
    parser.add_argument('modelpath',
        help = 'path to h5 model file')
    parser.add_argument('outputdir',
        help = 'path to submodels for shadownets')
    args = parser.parse_args()

    print(args.modelpath)
    print(args.outputdir)
    if not os.path.exists(args.outputdir):
            os.makedirs(args.outputdir)
    model = tf.keras.models.load_model(args.modelpath, custom_objects={'AddMask':AddMask,'LinearTransformGeneric':LinearTransformGeneric})
    convert_alexnet_to_shadownets(model, args.outputdir)
    #convert_mobilenet_to_darknet_models(model, 'mobilenet-submodels')
