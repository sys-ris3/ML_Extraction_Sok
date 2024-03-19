#!/usr/bin/env python3
from numpy import arange
import threading
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path
import tensorflow as tf
from os import path
import traceback
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel
from mobilenet_obf_custom_op import create_mobilenet_model_template
from mobilenet_model_converter import  mobilenet_converter_custom, mobilenet_converter_from_loaded_model
from evaluate_accuracy import evaluate_model_perf, evaluate_single_shadownet, get_eval_dic

"""
evaluation steps:
    1. obf ratio range(1, 1.1, 1.2, 1.3, ..., 2.0)
    2. For each obf ratio, generate model template
    3. For each model template, generate converted model weights
    4. For each converted model, evaluate it on image-net for accuracy, collect top-1, top-5 results
    5. For each converted model, evaluate it on image-net image for performance, average time on 100 images.

    results = [(obf_ratio, top-1, top-5, avg-time)]
"""

def thread_function_convert_model(index, obf_ratio, orig_model, results):
    try:
        obf_modelname = 'mobilenet_tpl_obf_ratio_' + str(obf_ratio).replace('.','_')+'.h5'
        converted_modelname = 'mobilenet_converted_obf_ratio_' + str(obf_ratio).replace('.','_')+'.h5'
        if not path.exists(obf_modelname):
            create_mobilenet_model_template(obf_modelname, obf_ratio)
        if not path.exists(converted_modelname):
        # generate model template
            obf_model = tf.keras.models.load_model(obf_modelname, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
            print("obf_model loaded %s" % obf_modelname)
            mobilenet_converter_from_loaded_model(orig_model, obf_model, converted_modelname, obf_ratio)
        results[index] = (obf_ratio, converted_modelname)
    except Exception as e:
        print("convert model at obf_ratio:%f failed!"%obf_ratio)
        print(e)
        traceback.print_exc()
        results[index] = None
    return

def generate_converted_models_sequential(obf_ratio_list):
    results = [None]*len(obf_ratio_list) 

    orig_modelname = 'mobilenet.h5'
    orig_model = tf.keras.models.load_model(orig_modelname)

    for index, obf_ratio in enumerate(obf_ratio_list):
        thread_function_convert_model(index, obf_ratio, orig_model, results)
    return results

def generate_converted_models(obf_ratio_list):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                                        datefmt="%H:%M:%S")


    results = [None]*len(obf_ratio_list) 
    threads = list()

    orig_modelname = 'mobilenet.h5'
    orig_model = tf.keras.models.load_model(orig_modelname)

    for index, obf_ratio in enumerate(obf_ratio_list):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function_convert_model, args=(index, obf_ratio, orig_model, results))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()

    return results

def evaluate_obf_ratio(obf_ratio_list, isTest):
    results = [] 

    fn_labels = "synsets.txt"
    fn_vals = "val.txt"
    label_dic = get_eval_dic(fn_labels, fn_vals)

    model_list = generate_converted_models_sequential(obf_ratio_list)
    datapath = "/root/mobilenet-eval/dataset/imagenet-val/"

    # original model 
    orig_modelname = 'mobilenet.h5'
    model_list.insert(0,(1.0, 'mobilenet.h5'))

    # other obfuscated model
    for obf_ratio, modelname in model_list:
        acc_result = evaluate_single_shadownet(modelname, datapath, label_dic, isTest)
        avg_time = evaluate_model_perf(modelname, datapath, 100)
        results.append((obf_ratio, acc_result[0], acc_result[1], avg_time))

    print (results)
    return results

def plot_results(results):
    obf_ratios = [r[0] for r in results]
    top_1 = [r[1] for r in results]
    top_5 = [r[2] for r in results]
    avg_time = [r[3] for r in results]
    max_time = max(avg_time)
    reg_avg_time = [x/max_time for x in avg_time] 

    fig, ax = plt.subplots()
    ax.plot(obf_ratios, top_1, label='top 1 acc')
    ax.plot(obf_ratios, top_5, label='top 5 acc')
    ax.plot(obf_ratios, reg_avg_time, label='regularized time')

    ax.set(xlabel='obfuscation ratio', ylabel='accuracy/regularized time',
                   title='Impact of ShadowNet obfuscation ratio on MobileNet')
    ax.grid()

    fig.savefig("obf_ratio_eval.png")
    plt.show()
    return

#test
def test_convert_model():
    print("unit test")
    results= [None]*2
    obf_ratio = 1.1

    orig_modelname = 'mobilenet.h5'
    orig_model = tf.keras.models.load_model(orig_modelname)
    
    thread_function_convert_model(0, obf_ratio, orig_model, results)

    print("test: generate converted model")

    fn_labels = "synsets.txt"
    fn_vals = "val.txt"
    label_dic = get_eval_dic(fn_labels, fn_vals)
    datapath = "/root/mobilenet-eval/dataset/imagenet-val/"

    acc_result = evaluate_single_shadownet(results[0][1], datapath, label_dic, True)
    avg_time = evaluate_model_perf(results[0][1], datapath, 100)

    print("test: evaluate acc and time")
    plot_results([(1.1, acc_result[0], acc_result[1],avg_time)])

    print("test: draw figure")
    return 

def eval_obf_ratio(isTest=True):
    obf_ratio_list = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    results = evaluate_obf_ratio(obf_ratio_list, isTest)
    plot_results(results)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t', '--unit-test', action='store_true',
                            help = 'regenerate report even if report is there')

    args = parser.parse_args()
    if args.unit_test is True:
        test_convert_model()
    else:
        eval_obf_ratio(isTest=False)
