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

def convert_minivgg_to_shadownets(model, outputdir):
    fh = ''
    for i in range(len(model.layers)):
        name = model.layers[i].name
        print("name:" + name)
        if name[:4] == 'conv':
            if name[:6] != 'conv_1':
                fh.close()
            fn = name[:4] + name[5]
            if fn == 'conv7':
                fn = 'results'
            fn += '.weights'
            print("file name:" + fn)
            fh = open(outputdir + '/' + fn, 'w+b')
            store_HD(fh)
        elif name[:4] == 'line':# linear_transform
            store_LT(fh, model, i)
        elif name[:4] == 'push' or name[:4] == 'pop_':
            store_AM(fh, model, i)
        elif name[:4] == 'batc':
            store_BN(fh, model, i)
        elif name[:4] == 'soft':
            fh.close()
        else:
            continue
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='minivgg_to_shadownet')
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
    convert_minivgg_to_shadownets(model, args.outputdir)
