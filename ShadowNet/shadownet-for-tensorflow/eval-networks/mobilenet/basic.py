#!/usr/bin/env python
from ctypes import *
import ctypes
import math
import random
import numpy as np
from evaluate_obf_enc_scheme_mobilenet import run_submodel 
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def darknet_predict(name, x):
    cfg_path = b"/root/darknet/cfg/mobilenet-subnets/" + name + b".cfg"
    weights_path = b"/root/darknet/data/mobilenet-submodels/" + name + b".weights"
    net = load_net(cfg_path, weights_path, 0)
    r = predict(net, x)
    return r
    
def print_bn_weights(layer):
    for weight in layer.weights:
        print("name:%s"%weight.name)
        print("%s %s %s %s"%(weight[0], weight[1],weight[2],weight[3]))
    return

def get_layers(cfg_tpl):
    m_type_dic = {"A": 5, "B":5,"C":6,"P":8,"R":5}
    num = m_type_dic[cfg_tpl[0]]
    start_layer = cfg_tpl[1]
    return range(start_layer, start_layer+num, 1)

if __name__ == "__main__":
    modelfile = 'mobilenet_obf_custom_filled.h5'
    model = tf.keras.models.load_model(modelfile, custom_objects={'AddMask':AddMask, 'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
    #model.summary()
    model_config = {b"conv1":("A",3),\
                    b"dwconv1":("B",9),\
                    b"pwconv1":("C",15),\
                    b"dwconv2":("B",23),\
                    b"pwconv2":("C",29),\
                    b"dwconv3":("B",36),\
                    b"pwconv3":("C",42),\
                    b"dwconv4":("B",50),\
                    b"pwconv4":("C",56),\
                    b"dwconv5":("B",63),\
                    b"pwconv5":("C",69),\
                    b"dwconv6":("B",77),\
                    b"pwconv6":("C",83),\
                    b"dwconv7":("B",90),\
                    b"pwconv7":("C",96),\
                    b"dwconv8":("B",103),\
                    b"pwconv8":("C",109),\
                    b"dwconv9":("B",116),\
                    b"pwconv9":("C",122),\
                    b"dwconv10":("B",129),\
                    b"pwconv10":("C",135),\
                    b"dwconv11":("B",142),\
                    b"pwconv11":("C",148),\
                    b"dwconv12":("B",156),\
                    b"pwconv12":("C",162),\
                    b"dwconv13":("B",169),\
                    b"pwconv13":("P",175),\
                    b"results":("R",184)}
    for m in model_config:
        input_shape = model.layers[model_config[m][1]].input_shape 
        layer_input = np.array(np.random.random_sample((1,) + input_shape[1:]), dtype=np.float32)
        layers = get_layers(model_config[m])
        tf_results = run_submodel(layers, model, layer_input) 
        darknet_results = darknet_predict(m,layer_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        print("m:%s"%m)
        if (len(tf_results.shape) == 4):
            print("results tf:%s , darknet:%s" %(tf_results[0][0][0][0], darknet_results[0])) 
        else:
            print("results tf:%s , darknet:%s" %(tf_results[0][0], darknet_results[0])) 

