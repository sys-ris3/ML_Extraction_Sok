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
def debug(m_ori,m_obf,quantize):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    model_ori = tf.keras.models.load_model(m_ori,  custom_objects={"ResNetBlock":ResNetBlock})
    model_obf = tf.keras.models.load_model(m_obf, custom_objects={"ResNetBlock":ResNetBlock,'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})
    to_eval_img = x_test
    to_eval_label = y_test
    total = to_eval_img.shape[0]
    eval_img = to_eval_img.astype('float32') / 255
    
    eval_img = np.reshape(eval_img[0], (1,) + eval_img[0].shape)
    shadow_index = 4
    orig_index = 4
    
    
    x_s = eval_img * 256
    x_o = eval_img
    for i in range(1,shadow_index):
        x_s = model_obf.layers[i](x_s)
    for i in range(1,orig_index):
        x_o = model_ori.layers[i](x_o)
    assert(isinstance(model_obf.layers[shadow_index],ResNetBlock))
    assert(isinstance(model_ori.layers[orig_index],ResNetBlock))
    for l in model_obf.layers[4].layer1_functors:
        x_s = l(x_s)
    for l in model_ori.layers[4].layer1_functors:
        x_o = l(x_o)
    for l in model_obf.layers[4].layer2_functors:
        x_s = l(x_s)
    for l in model_ori.layers[4].layer2_functors:
        x_o = l(x_o)
    
    embed()
def compare(m_ori, m_obf, quantize=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    model_ori = tf.keras.models.load_model(m_ori,  custom_objects={"ResNetBlock":ResNetBlock})
    model_obf = tf.keras.models.load_model(m_obf, custom_objects={"ResNetBlock":ResNetBlock,'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})
    #printf("ori_loss:%f,ori_acc:%f, obf_loss:%f,obf_acc:%f" % (ori_loss, ori_acc, obf_loss, obf_acc))
    matched_1 = 0
    matched_2 = 0
    orig_acc_top1 = 0
    obf_acc_top1 = 0
    orig_acc_top5 = 0
    obf_acc_top5 = 0
    total = x_test.shape[0]
    eval_img = x_test
    

    eval_y = y_test
    for i in range(total):
        if i% 100 == 0:
            print(i)
            print("total:{}\ntested:{}\norig_acc_top1:{}\nobf_acc_top1:{}\norig_acc_top5:{}\nobf_acc_top5:{}\nmatched_1:{}".format(total,i,orig_acc_top1,obf_acc_top1,orig_acc_top5,obf_acc_top5,matched_1))
        
        test1 = np.reshape(eval_img[i], (1,) + eval_img[i].shape)
        
        if quantize:
            test1q = test1.astype('float32') * 256.0
        else:
            test1q = test1
        ori = model_ori.predict(test1)
        obf = model_obf.predict(test1q)

        orig_p = np.argsort(ori, axis=1)
        obf_p = np.argsort(obf, axis=1)
        gt = eval_y[i][0]

        obf_top2 = obf_p[0][-2:]
        orig_top2 =  orig_p[0][-2:]
        
        
        if orig_p[0][-1] == gt:
            orig_acc_top1 += 1
        if obf_p[0][-1] == gt:
            obf_acc_top1 += 1
        if gt in orig_p[0][-5:]:
            orig_acc_top5 += 1
        if gt in obf_p[0][-5:]:
            obf_acc_top5 += 1
        if obf_p[0][-1] == orig_p[0][-1]:
            matched_1 += 1
    print("total:{}\ntested:{}\norig_acc_top1:{}\nobf_acc_top1:{}\norig_acc_top5:{}\nobf_acc_top5:{}\nmatched_1:{}".format(total,i,orig_acc_top1,obf_acc_top1,orig_acc_top5,obf_acc_top5,matched_1))

    
if __name__ == '__main__':
    '''
    m_ori = "trained_resnet-44_aug.h5"
    quant = True
    if quant:
        m_obfq = "trained_resnet-44_aug_auto_obf_quant.h5"
    else:
        m_obfq = "trained_resnet-44_aug_auto_obf.h5"
    '''
    m_ori = "trained_resnet-404_aug.h5"
    quant = True
    if quant:
        m_obfq = "trained_resnet-404_aug_auto_obf_quant.h5"
    else:
        m_obfq = "trained_resnet-404_aug_auto_obf.h5"
    
    compare(m_ori, m_obfq, quantize=quant)