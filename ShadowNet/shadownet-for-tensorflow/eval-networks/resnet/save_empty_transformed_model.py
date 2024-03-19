from resnet import resnet_v1,resnet_v1_obf
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow.keras
import json

checkpoint_path = "training_2/cp-0123.ckpt"

(train_imgs, train_labels), (test_imgs, test_labels) = cifar10.load_data()
# Training parameters
BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128
EPOCHS = 200 # 200
USE_AUGMENTATION = True
NUM_CLASSES = np.unique(train_labels).shape[0] # 10
COLORS = train_imgs.shape[3]
input_shape = train_imgs.shape[1:]
SUBTRACT_PIXEL_MEAN = True
VERSION = 1

DEPTH = 44

test_imgs = test_imgs.astype('float32') / 255
test_labels = tensorflow.keras.utils.to_categorical(test_labels, NUM_CLASSES)


model = resnet_v1_obf(input_shape=input_shape, depth=DEPTH)

## prepare to train
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr
model.compile(loss='categorical_crossentropy',
          optimizer=Adam(lr=lr_schedule(0)),
          metrics=['accuracy'])


model.save('resnet-{}-empty-transformed.h5'.format(DEPTH))
with open("resnet-{}-empty-transformed.info".format(DEPTH),"w") as wfile:
    model.summary(print_fn=lambda x:wfile.write(x+'\n'))

conv_input_shape_dict = {}
for each_layer in model.layers:
    if 'conv2d' in each_layer.name:
        conv_input_shape_dict[each_layer.name] = list(each_layer.input.shape)
with open("obf_model_conv_input_dict.json",'w') as wfile:
    json.dump(conv_input_shape_dict,wfile)

plot_model(model, to_file='model-transformed.png')
'''
scores = model.evaluate(test_imgs, test_labels, verbose=1)
    
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
'''
