from resnet import resnet_v1,resnet_v1_obf
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow.keras

checkpoint_path = "training_2/cp-0146.ckpt"

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


model = resnet_v1(input_shape=input_shape, depth=DEPTH)

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

model.load_weights(checkpoint_path)

scores = model.evaluate(test_imgs, test_labels, verbose=1)
print(scores)

## model.save('resnet-{}.h5'.format(DEPTH))

# plot_model(model, to_file='model.png')
'''
scores = model.evaluate(test_imgs, test_labels, verbose=1)
    
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
'''
