import sys
import tensorflow
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

def compare(m_ori, m_obf, quantize=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    model_ori = tensorflow.keras.models.load_model(m_ori,  custom_objects={"ResNetBlock":ResNetBlock})
    print(model_ori.summary())
    model_obf = tensorflow.keras.models.load_model(m_obf, custom_objects={"ResNetBlock":ResNetBlock,'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})
    #printf("ori_loss:%f,ori_acc:%f, obf_loss:%f,obf_acc:%f" % (ori_loss, ori_acc, obf_loss, obf_acc))
    
    matched = 0
    orig_acc_top1 = 0
    obf_acc_top1 = 0
    orig_acc_top5 = 0
    obf_acc_top5 = 0
    total = x_test.shape[0]
    for i in range(total):
        if i% 100 == 0:
            print(i)
            print("total tested:{}\norig_acc_top1:{}\nobf_acc_top1:{}\norig_acc_top5:{}\nobf_acc_top5:{}\nmatched:{}".format(total,orig_acc_top1,obf_acc_top1,orig_acc_top5,obf_acc_top5,matched))
        test1 = np.reshape(x_test[i], (1,) + x_test[i].shape)
        test1q = test1.astype('float32')*256.0
        ori = model_ori.predict(test1)
        obf = model_obf.predict(test1q)

        orig_p = np.argsort(ori, axis=1)
        obf_p = np.argsort(obf, axis=1)
        gt = y_test[i][0]
        gt_wrong = np.argmax(y_test[i])
        if(gt_wrong != 0):
            assert(False)
        if orig_p[0][-1] == gt:
            orig_acc_top1 += 1
        if obf_p[0][-1] == gt:
            obf_acc_top1 += 1
        if gt in orig_p[0][-5:]:
            orig_acc_top5 += 1
        if gt in obf_p[0][-5:]:
            obf_acc_top5 += 1
        if obf_p[0][-1] == orig_p[0][-1]:
            matched += 1
    print("total tested:{}\norig_acc_top1:{}\nobf_acc_top1:{}\norig_acc_top5:{}\nobf_acc_top5:{}\nmatched:{}".format(total,orig_acc_top1,obf_acc_top1,orig_acc_top5,obf_acc_top5,matched))

    
if __name__ == '__main__':
    tensorflow.config.threading.set_inter_op_parallelism_threads(8)
    tensorflow.config.threading.set_intra_op_parallelism_threads(8)
    m_ori = "trained_minivgg_77.h5"
    m_obfq = "trained_minivgg_77_auto_obf_quant.h5"
    compare(m_ori, m_obfq, quantize=True)
