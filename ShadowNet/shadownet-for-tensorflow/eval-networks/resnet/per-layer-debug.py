from resnet import resnet_v1,resnet_v1_obf
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import Model

from add_mask_layer import AddMask
#from linear_transform_layer import LinearTransform
from linear_transform_generic_layer import LinearTransformGeneric
from shuffle_channel_layer import ShuffleChannel
from IPython import embed
np.set_printoptions(suppress=True)

(_, _), (test_imgs, test_labels) = cifar10.load_data()
# Training parameters
BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128
EPOCHS = 200 # 200
USE_AUGMENTATION = True
NUM_CLASSES = np.unique(test_labels).shape[0] # 10
COLORS = test_imgs.shape[3]
input_shape = test_imgs.shape[1:]
SUBTRACT_PIXEL_MEAN = True
VERSION = 1

DEPTH = 44

test_imgs = test_imgs.astype('float32') / 255
test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)


def main(orig_model,obf_model,orig_layer_name,obf_layer_name,orig_layer_num,obf_layer_num):
    
    
    partial_orig_model = Model(orig_model.input,orig_model.layers[orig_layer_num].output)
    partial_obf_model = Model(obf_model.input, obf_model.layers[obf_layer_num].output)

    with open("orig_dbg_out.txt","a") as wfile:
        wfile.write(orig_layer_name+'\n')
        wfile.write(str(partial_orig_model.predict(test_imgs[0:1]))+'\n')
    with open("obf_dbg_out.txt","a") as wfile:
        wfile.write(obf_layer_name+'\n')
        wfile.write(str(partial_obf_model.predict(test_imgs[0:1]))+'\n')
    
if __name__ == '__main__':
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    
    orig_model_path = './resnet-44.h5'
    obf_model_path = './resnet-44-trans-filled.h5'
    orig_model = tf.keras.models.load_model(orig_model_path)
    obf_model = tf.keras.models.load_model(obf_model_path,custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ShuffleChannel':ShuffleChannel})
    names = [('add_20','add_20'),('activation_42','activation_42'),('average_pooling2d','average_pooling2d'),('flatten','flatten'),('dense','dense')]
    for orig_layer_name,obf_layer_name in names:
        '''
    for add_idx in range(1,21):
        orig_layer_name = "add_{}".format(add_idx)
        obf_layer_name = "add_{}".format(add_idx)
        '''
        orig_layer_num = None
        obf_layer_num = None

        for idx, each_layer in enumerate(orig_model.layers):
            if each_layer.name == orig_layer_name:
                orig_layer_num = idx
                break
        assert(orig_layer_num != None)

        for idx, each_layer in enumerate(obf_model.layers):
            if each_layer.name == obf_layer_name:
                obf_layer_num = idx
                break
        assert(obf_layer_num != None)
        print(orig_layer_name,obf_layer_name)
        main(orig_model,obf_model,orig_layer_name,obf_layer_name,orig_layer_num,obf_layer_num)
