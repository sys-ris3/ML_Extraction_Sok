#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel
import random
from matplotlib import pyplot as plt

IMG_H = 224
IMG_W = 224

"""
Compare tensor a and b of shape(B, H, W, C) elementwisely 
and return the accumulated errors.
"""
def compare_tensor(a, b):
    print("a, b shapes")
    print(a.shape)
    print(b.shape)

    # store accumulated errors on all tensor elements
    errs = 0
    err_rs = []
    a_vals = 0 

    if a.shape != b.shape:
        print ("two tensors are in different shape")
        return

    # randomly pick some tensor to compare
    for cnt in range(10):
        i = 0 
        j =  random.randint(0, a.shape[1] - 1)
        k =  random.randint(0, a.shape[2] - 1)
        c =  random.randint(0, a.shape[3] - 1)
        err = a[i][j][k][c] - b[i][j][k][c]
        if err > 0.00001: # report on big error
            print ("i:%d, j:%d, k:%d, err:%f"%(i,j,k,err))

        errs += err if (err > 0) else (-1.0 * err)

        # error ratio : err / val
        ea = err/a[i][j][k][c]
        eb = err/b[i][j][k][c]

        # original value of a/model
        av = a[i][j][k][c].numpy()
        if av > 0:
            a_vals += av 
        else:
            a_vals += (-1.) * av


        # report middle result because it may take long to finish
        err_rs.append((ea.numpy(), eb.numpy()))

    avg_a = a_vals/10.0
    print ("errs:%f, avg_err:%f"%(errs, errs/10.0))
    print ("err ratios:")
    print (err_rs)
    print ("original model output:")
    print (a_vals)
    return errs, err_rs, avg_a

def get_avg_err_ratio(err_rs):
    ea = 0.
    eb = 0.
    for e in err_rs:
        ea += e[0] if e[0] >0.0 else (e[0] * (-1.0)) 
        eb += e[1] if e[1] >0.0 else (e[1] * (-1.0)) 
    return (ea/len(err_rs), eb/len(err_rs))

"""
A fast version of compare_tensor, needs hardware support
for tf.reduce_sum() 
"""
def compare_tensor_fast(a, b):
    assert a.shape == b.shape
    # not supported so far
    print(tf.reduce_sum(a, -b))

"""
Get a submodel from the original model with designated layers
"""
def run_submodel(layers, model, inputs):
    x = inputs
    for i in layers:
        x = model.layers[i](x)
        #if len(x.shape) == 4:
        #    print("x[0]:%s, x[1]:%s"%(x[0][0][0][0],x[0][0][0][1]))
        #else:
        #    print("x[0]:%s, x[1]:%s"%(x[0][0],x[0][1]))
    return x

"""
Evaluate designated layers of a model against an obfuscated one.
"""
def evaluate_model_periods(model, obf_model, mp, omp, inputs):
    mo = run_submodel(mp, model, inputs)
    omo = run_submodel(omp, obf_model, inputs)
    return compare_tensor(mo, b=omo)


"""
Evaluate whether the obfuscated model is equalvent with 
original model by operating on random input. 
"""
def evaluate_obfuscated_encrypted_model(orig_model, obf_enc_model):
    # load original model
    model = tf.keras.models.load_model(orig_model)
    #model.summary()
    
    # load obfuscated model template
    #oe_model = tf.keras.models.load_model(obf_enc_model, custom_objects={'AddMask':AddMask})
    oe_model = tf.keras.models.load_model(obf_enc_model, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    #oe_model.summary()

    layer_eratio_a= []
    layer_eratio_b= []
    acc_eratio_a= []
    acc_eratio_b= []
    avg_as = []
    acc_avg_as = []

    # evaluate the first layer
    print("this layer 0 error:")
    img_input = tf.random.uniform((1,IMG_H, IMG_W, 3))
    mp = range(0,4)
    omp = range(0,5)
    errs, err_rs, avg_a = evaluate_model_periods(model, oe_model, mp, omp, img_input)
    print("")

    layer_eratio_a.append(get_avg_err_ratio(err_rs)[0])
    layer_eratio_b.append(get_avg_err_ratio(err_rs)[1])
    acc_eratio_a.append(get_avg_err_ratio(err_rs)[0])
    acc_eratio_b.append(get_avg_err_ratio(err_rs)[1])
    avg_as.append(avg_a)
    acc_avg_as.append(avg_a)

    input_layers = [5, 11, 18, 24, 31, 37, 44, 50, 56, 62, 68, 74, 81, 89]

    mps = [(5,10),(11, 17), (18, 23), (24, 30), (31, 36), (37,43), (44, 49),(50, 55),(56, 61),(62, 67),(68, 73),(74,80),(81, 86),(89,91)]
    
    omps = [(6,18),(19, 32), (33, 45), (46, 59), (60, 72), (73, 86), (87,99), (100,112),(113,125),(126,138), (139,151), (152, 165), (166,178),(181,187)]

    
    for i in range(len(input_layers)):
        # evaluate i layer
        print("this layer %d error:"%(i+1))
        input_shape = model.layers[input_layers[i]].input_shape 
        layer_input = tf.random.uniform((1,) + input_shape[1:])
        print ("img_input.shape")
        print (layer_input.shape)
        mp = range(mps[i][0],mps[i][1])
        omp = range(omps[i][0], omps[i][1])
        errs, err_rs, avg_a = evaluate_model_periods(model, oe_model, mp, omp, layer_input)
        layer_eratio_a.append(get_avg_err_ratio(err_rs)[0])
        layer_eratio_b.append(get_avg_err_ratio(err_rs)[1])
        avg_as.append(avg_a)
    
        # evaluaete 0 - i layer
        print("")
        print("accumulated errors from 0 to layer %d:"%(i+1))
        img_input = tf.random.uniform((1,IMG_H, IMG_W, 3))
        mp = range(0,mps[i][1])
        omp = range(0, omps[i][1])
        errs, err_rs, avg_a = evaluate_model_periods(model, oe_model, mp, omp, img_input)
        print("")
        acc_eratio_a.append(get_avg_err_ratio(err_rs)[0])
        acc_eratio_b.append(get_avg_err_ratio(err_rs)[1])
        acc_avg_as.append(avg_a)
    
    return (layer_eratio_a, layer_eratio_b, acc_eratio_a, acc_eratio_b, avg_as,acc_avg_as)

if __name__ == '__main__':
    orig_model = 'mobilenet.h5'
    obf_enc_model = 'mobilenet_obf_custom_filled.h5'
    #orig_model = 'original/mobilenet.h5'
    #obf_enc_model = 'original/mobilenet_obf_filled.h5'
    #orig_model = 'zero-mask/mobilenet.h5'
    #obf_enc_model = 'zero-mask/mobilenet_obf_filled.h5'
    (la,lb,aa,ab,avga,acca) = evaluate_obfuscated_encrypted_model(orig_model, obf_enc_model)

    plt.xlabel('layer id')
    plt.ylabel('average error ratio')
    plt.plot(la, 'gs', lw=2, label='layer(err/model)') 
    plt.plot(aa, 'bo', lw=2, label='acc(err/model)')
    plt.plot(lb, 'y+', lw=2, label='layer(err/obf)') 
    plt.plot(ab, 'r+', lw=2, label='acc(err/obf)')
    plt.plot(avga,'go', lw=2, label='layer(output)')
    plt.plot(acca,'rs', lw=2, label='acc(output)')

    plt.yscale('log')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .402), loc='lower left',
                       ncol=3, mode="expand", borderaxespad=0.)
    #plt.show()
    print("save errs.png")
    plt.savefig('errs.png')
