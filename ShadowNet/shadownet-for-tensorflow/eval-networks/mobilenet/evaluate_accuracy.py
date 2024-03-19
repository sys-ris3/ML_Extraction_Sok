#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from linear_transform_generic_layer import LinearTransformGeneric
from shuffle_channel_layer import ShuffleChannel
import random
import traceback
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.layers import Layer,Conv2D,Dense,InputLayer,AveragePooling2D,MaxPooling2D,Activation,Flatten,Reshape,Dropout,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,Lambda,DepthwiseConv2D,ReLU, Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.nn import relu6
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )
from os import listdir
from os.path import isfile, join
import logging
import threading
import time
from IPython import embed

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



MAX_THREADS = 16 
def get_inputlist(datapath):
    return [join(datapath, f) for f in listdir(datapath) if isfile(join(datapath, f))]

def run_model_on_image(image_file, model):
    orig_img = load_img(image_file, target_size=(224, 224))
    numpy_image = img_to_array(orig_img)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    try:
        processed_image = mobilenet.preprocess_input(image_batch.copy())
        start_time = time.time()
        orig_pred = model.predict(processed_image)
        end_time = time.time()
        duration = end_time - start_time
    except Exception as e:
        print("error happened when process %s, skip"%image_file)
        print(e)
        traceback.print_exc()
        orig_pred = None
        duration = 0

    return duration 


def thread_function(orig_model, obf_model, image_list, label_dic, quantize):
    nums = len(image_list)
    orig_top_1 = 0
    orig_top_5 = 0
    obf_top_1 = 0
    obf_top_5 = 0
    matched = 0
    num_images = len(image_list)
    i = 0
    matched_prob = 0.0
    unmatched_prob = 0.0
    #orig_correct_consistency = 0
    occ = 0
    #orig_incorrect_consistency = 0
    oic = 0
    for image_file in image_list:
        i += 1
        if (i%100 == 0):
            print(f" i:{i}\n o_top_1:{orig_top_1}\n s_top_1:{obf_top_1}\n o_top_5:{orig_top_5}\n s_top_5:{obf_top_5}\n matched:{matched}") 
            print("orig_correct_consistency{}".format(occ/orig_top_1))
            print("orig_incorrect_consistency{}".format(oic/(i - orig_top_1)))
            print("avg_matched_prob:{}".format(matched_prob/matched))
            print("avg_unmatched_prob:{}".format(unmatched_prob/(i - matched)))
        # preprocess image
        orig_img = load_img(image_file, target_size=(224, 224))
        numpy_image = img_to_array(orig_img)
        image_batch = np.expand_dims(numpy_image, axis=0)

        try:
            processed_image = mobilenet.preprocess_input(image_batch.copy())
            if quantize:
                processed_image_q = processed_image * (256.)
            orig_pred = orig_model.predict(processed_image)
            if quantize:
                obf_pred = obf_model.predict(processed_image_q)
            else:
                obf_pred = obf_model.predict(processed_image)
        except:
            print("error happened when process %s, skip"%image_file)
            continue

        # parse results
        label_orig_pred = decode_predictions(orig_pred)
        label_obf_pred = decode_predictions(obf_pred)

        image_fn = image_file.split('/')[-1].strip()
        target_label = label_dic[image_fn]

        if label_orig_pred[0][0][0] == target_label:
            orig_top_1 += 1
            if label_obf_pred[0][0][0] == label_orig_pred[0][0][0]:
                occ += 1
        else:
            if label_obf_pred[0][0][0] == label_orig_pred[0][0][0]:
                oic += 1

        if label_obf_pred[0][0][0] == target_label:
            obf_top_1 += 1
        if label_obf_pred[0][0][0] == label_orig_pred[0][0][0]:
            matched += 1
            matched_prob += label_orig_pred[0][0][2]
        else:
            unmatched_prob += label_orig_pred[0][0][2]

        #else:
            #print("label_obf_pred:")
            #print(label_obf_pred)
            #print("label_orig_pred:")
            #print(label_orig_pred)
            #embed()

        top_5_labels = [x[0] for x in label_orig_pred[0]] 
        #print(top_5_labels)
        if target_label in top_5_labels:
            orig_top_5 += 1

        top_5_labels = [x[0] for x in label_obf_pred[0]] 
        #print(top_5_labels)
        if target_label in top_5_labels:
            obf_top_5 += 1

    print("orig_correct_consistency{}".format(occ/orig_top_1))
    print("orig_incorrect_consistency{}".format(oic/(num_images - orig_top_1)))
    print("avg_matched_prob:{}".format(matched_prob/matched))
    print("avg_unmatched_prob:{}".format(unmatched_prob/(num_images - matched)))

    return (orig_top_1,orig_top_5,obf_top_1, obf_top_5, matched, num_images)

def thread_function_one_model(index, obf_model, image_list, label_dic, results):
    obf_top_1 = 0
    obf_top_5 = 0
    num_images = len(image_list)
    for image_file in image_list:
        # preprocess image
        orig_img = load_img(image_file, target_size=(224, 224))
        numpy_image = img_to_array(orig_img)
        image_batch = np.expand_dims(numpy_image, axis=0)

        try:
            processed_image = mobilenet.preprocess_input(image_batch.copy())
            obf_pred = obf_model.predict(processed_image)
        except:
            print("error happened when process %s, skip"%image_file)
            continue

        # parse results
        label_obf_pred = decode_predictions(obf_pred)

        image_fn = image_file.split('/')[-1].strip()
        target_label = label_dic[image_fn]

        if label_obf_pred[0][0][0] == target_label:
            obf_top_1 += 1

        top_5_labels = [x[0] for x in label_obf_pred[0]] 
        #print(top_5_labels)
        if target_label in top_5_labels:
            obf_top_5 += 1


    results[index] = (obf_top_1, obf_top_5, num_images)
    return
     
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]
	
def evaluate_shadownet(orig_model, obf_model, datapath, label_dic, quantize):
    inputlist = get_inputlist(datapath)

    # load two models
    orig_model_ld = tf.keras.models.load_model(orig_model)
    obf_model_ld = tf.keras.models.load_model(obf_model, custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ShuffleChannel':ShuffleChannel, 'ActivationQ':ActivationQ, 'LinearTransform':LinearTransform})

    r = thread_function(orig_model_ld, obf_model_ld, inputlist, label_dic, quantize)

    orig_top_1 = r[0] 
    orig_top_5 = r[1] 
    obf_top_1 = r[2] 
    obf_top_5 = r[3] 
    matched = r[4]
    num_images = r[5]
    print(orig_top_1,orig_top_5,obf_top_1, obf_top_5, matched, num_images)
    print("orig model top 1:%f"%(orig_top_1/float(num_images)))
    print("obf model top 1:%f"%(obf_top_1/float(num_images)))
    print("orig model top 5:%f"%(orig_top_5/float(num_images)))
    print("obf model top 5:%f"%(obf_top_5/float(num_images)))
    print("consistency: :%f"%(matched/float(num_images)))
    return

"""
labels:
    ...
    ILSVRC2012_val_00000013.JPEG 370
    ...
vals:
    ...
    n04141975
    ...
returns dic {ILSVRC2012_val_00000013.JPEG:n04141975}
"""
def get_eval_dic(fn_labels, fn_vals):
    labels = open(fn_labels,'r').readlines()
    vals = open(fn_vals,'r').readlines()
    label_dic = {}
    for v in vals:
        fields = v.split()
        if len(fields) == 2:
            image, index = fields[0].strip(), int(fields[1].strip())
            label_dic[image] = labels[index].strip()
    print([*label_dic][1])
    print(label_dic[[*label_dic][1]])
    return label_dic

"""
evaluate obf_model on num images from datapath
return average model inference time
"""
def evaluate_model_perf(obf_model, datapath, num):
    inputlist = get_inputlist(datapath)[:num]
    obf_model_ld = tf.keras.models.load_model(obf_model, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    total_time = 0
    N = len(inputlist)
    for image_file in inputlist: 
        total_time +=  run_model_on_image(image_file, obf_model_ld)
    return total_time/float(N)

def evaluate_single_shadownet(obf_model, datapath, label_dic, isTest):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                                        datefmt="%H:%M:%S")
    # enumerate data filenames under datapath
    if isTest:
        inputlist = get_inputlist(datapath)[:32]
    else:
        inputlist = get_inputlist(datapath)

    N = len(inputlist) 
    chunk_size = N // MAX_THREADS
    slices = chunks(inputlist, chunk_size)

    # load two models
    obf_model_ld = tf.keras.models.load_model(obf_model, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})

    threads = list()
    results = [None] * len(slices)
    for index, s in enumerate(slices):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function_one_model, args=(index, obf_model_ld, s, label_dic, results))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        #logging.info("Main    : thread %d done", index)        

    obf_top_1 = 0
    obf_top_5 = 0 
    num_images = 0

    for r in results:
        if r is None:
            print("r is None! skip")
            continue
        obf_top_1 += r[0] 
        obf_top_5 += r[1] 
        num_images += r[2]
    print(obf_top_1, obf_top_5, num_images)
    print("obf model top 1:%f"%(obf_top_1/float(num_images)))
    print("obf model top 5:%f"%(obf_top_5/float(num_images)))
    return (obf_top_1/float(num_images), obf_top_5/float(num_images))

"""
Input: 50,000 evaluation images and two models
Output: rate: the percentage of images that two models agree on each other.
"""
if __name__ == '__main__':
    orig_model = 'mobilenet.h5'
    #obf_model = 'mobilenet_obf_custom_filled.h5'
    obf_model = 'mobilenet_auto_obf_quant_new.h5'
    datapath = "/root/mobilenet-eval/dataset/imagenet-val/"
    fn_labels = "synsets.txt"
    fn_vals = "val.txt"
    quantize = True

    label_dic = get_eval_dic(fn_labels, fn_vals)
    evaluate_shadownet(orig_model, obf_model, datapath, label_dic, quantize)
