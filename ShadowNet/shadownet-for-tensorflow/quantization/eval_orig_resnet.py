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
import os

from tensorflow.keras.optimizers import SGD

from IPython import embed


cur_dir = os.path.dirname(__file__)
resnet_package_path = os.path.join(cur_dir,"..","eval-networks","resnet")
sys.path.append(resnet_package_path)

from resnet import ResNetBlock

def compare(m_ori):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    model_ori = tf.keras.models.load_model(m_ori,  custom_objects={"ResNetBlock":ResNetBlock})

    model_ori.compile(loss='categorical_crossentropy', 
        optimizer=SGD(lr=0.001), metrics=['accuracy'])
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    scores = model_ori.evaluate(x_test, y_test, verbose=1)
    
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
if __name__ == '__main__':
    m_ori = "../eval-networks/resnet/resnet-new-44.h5"
    compare(m_ori)
