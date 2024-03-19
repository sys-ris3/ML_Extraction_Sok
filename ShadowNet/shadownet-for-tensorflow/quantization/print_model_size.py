from statistics import mode
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric

from tensorflow.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Reshape
from quant_transform import ActivationQ
import os
from IPython import embed

cur_dir = os.path.dirname(__file__)
resnet_package_path = os.path.join(cur_dir,"..","eval-networks","resnet")
sys.path.append(resnet_package_path)

from resnet import ResNetBlock


model_ori = tf.keras.models.load_model("trained_resnet-404_aug.h5",  custom_objects={"ResNetBlock":ResNetBlock})

model_obf = tf.keras.models.load_model("trained_resnet-404_aug_auto_obf_quant.h5", custom_objects={"ResNetBlock":ResNetBlock,'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})

orig_conv = 0
orig_other = 0

trans_conv = 0
trans_other = 0
with open("resnet404_size.txt","w") as wfile:
    for each_layer in model_ori.layers:
        if(isinstance(each_layer, ResNetBlock)):
            for cur_layer in each_layer.layer1_functors:
                wfile.write("{},layer1,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                if(isinstance(cur_layer, Conv2D)):
                    orig_conv += cur_layer.count_params()
                else:
                    orig_other += cur_layer.count_params()
            for cur_layer in each_layer.layer2_functors:
                wfile.write("{},layer2,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                if(isinstance(cur_layer, Conv2D)):
                    orig_conv += cur_layer.count_params()
                else:
                    orig_other += cur_layer.count_params()
            if(hasattr(each_layer, 'layer3_functors')):
                for cur_layer in each_layer.layer3_functors:
                    wfile.write("{},layer3,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                    if(isinstance(cur_layer, Conv2D)):
                        orig_conv += cur_layer.count_params()
                    else:
                        orig_other += cur_layer.count_params()
            for cur_layer in each_layer.layer4_functors[1:]:
                wfile.write("{},layer4,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                if(isinstance(cur_layer, Conv2D)):
                    orig_conv += cur_layer.count_params()
                else:
                    orig_other += cur_layer.count_params()
        else:
            wfile.write("{}:{}\n".format(each_layer.name,each_layer.count_params()))
            if(isinstance(each_layer, Conv2D)):
                orig_conv += each_layer.count_params()
            else:
                orig_other += each_layer.count_params()
    wfile.write("\n\n")
    for each_layer in model_obf.layers:
        if(isinstance(each_layer, ResNetBlock)):
            for cur_layer in each_layer.layer1_functors:
                wfile.write("{},layer1,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                if(isinstance(cur_layer, Conv2D)):
                    trans_conv += cur_layer.count_params()
                else:
                    trans_other += cur_layer.count_params()
            for cur_layer in each_layer.layer2_functors:
                wfile.write("{},layer2,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                if(isinstance(cur_layer, Conv2D)):
                    trans_conv += cur_layer.count_params()
                else:
                    trans_other += cur_layer.count_params()
            if(hasattr(each_layer, 'layer3_functors')):
                for cur_layer in each_layer.layer3_functors:
                    wfile.write("{},layer3,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                    if(isinstance(cur_layer, Conv2D)):
                        trans_conv += cur_layer.count_params()
                    else:
                        trans_other += cur_layer.count_params()
            for cur_layer in each_layer.layer4_functors[1:]:
                wfile.write("{},layer4,{}:{}\n".format(each_layer.name,cur_layer.name,cur_layer.count_params()))
                if(isinstance(cur_layer, Conv2D)):
                    trans_conv += cur_layer.count_params()
                else:
                    trans_other += cur_layer.count_params()
        else:
            wfile.write("{}:{}\n".format(each_layer.name,each_layer.count_params()))
            if(isinstance(each_layer, Conv2D)):
                trans_conv += each_layer.count_params()
            else:
                trans_other += each_layer.count_params()
orig_conv = orig_conv * 4 / 1024 / 1024 
orig_other = orig_other * 4 / 1024 / 1024 
trans_conv = trans_conv * 4 / 1024 / 1024 
trans_other = trans_other * 4 / 1024 / 1024 

print("orig_conv:{},orig_other:{}\ntrans_conv:{},trans_other:{}\n".format(orig_conv,orig_other, trans_conv,trans_other))