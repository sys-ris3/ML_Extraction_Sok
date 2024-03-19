from resnet import resnet_v1,resnet_v1_obf
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import tensorflow.keras

from tqdm import tqdm

from add_mask_layer import AddMask
#from linear_transform_layer import LinearTransform
from linear_transform_generic_layer import LinearTransformGeneric
from shuffle_channel_layer import ShuffleChannel
from IPython import embed

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
test_labels = tensorflow.keras.utils.to_categorical(test_labels, NUM_CLASSES)


def main(orig_model_path,obf_model_path):
    orig_model = tf.keras.models.load_model(orig_model_path)
    obf_model = tf.keras.models.load_model(obf_model_path,custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ShuffleChannel':ShuffleChannel})
    matched = 0
    orig_acc_top1 = 0
    obf_acc_top1 = 0

    orig_acc_top5 = 0
    obf_acc_top5 = 0
    total = test_imgs.shape[0]
    for img_idx in tqdm(range(total)):
        orig_p_dis = orig_model.predict(test_imgs[img_idx:img_idx+1])
        obf_p_dis = obf_model.predict(test_imgs[img_idx:img_idx+1])
        orig_p = np.argsort(orig_p_dis,axis=1)
        obf_p = np.argsort(obf_p_dis,axis=1)
        gt = np.argmax(test_labels[img_idx])
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
    
    orig_model_path = './resnet-44.h5'
    obf_model_path = './resnet-44-trans-filled.h5'

    main(orig_model_path,obf_model_path)
