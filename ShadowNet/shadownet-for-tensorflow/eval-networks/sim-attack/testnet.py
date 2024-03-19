import tensorflow
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine import training
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import random as rand

(train_imgs, train_labels), (test_imgs, test_labels) = cifar10.load_data()

def original_net(input_shape, layer_num, num_classes=10):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for i in range(layer_num):
        model.add(layers.Conv2D(64, 3, 1, activation="relu"))
        model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=8))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes,
        activation='softmax',
        kernel_initializer='he_normal'))
    #model.layers[0].trainable = False

    return model

def train(epochs, layer_num):
    global train_imgs,train_labels,test_imgs,test_labels 
    # Training parameters
    BATCH_SIZE = 128  
    EPOCHS = epochs # 200
    USE_AUGMENTATION = True
    NUM_CLASSES = np.unique(train_labels).shape[0] # 10
    COLORS = train_imgs.shape[3]
    input_shape = train_imgs.shape[1:]
    SUBTRACT_PIXEL_MEAN = True
    checkpoint_path = "training_test/cp-{epoch:04d}.ckpt"
    VERSION = 1


    # Normalize data.
    train_imgs = train_imgs.astype('float32') / 255
    test_imgs = test_imgs.astype('float32') / 255
    train_labels = tensorflow.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = tensorflow.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    model = original_net(input_shape, layer_num)


    ## prepare to train
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    
    #%model.compile(loss='categorical_crossentropy', 
    #%    optimizer=SGD(lr=0.001), metrics=['accuracy'])
    #        optimizer=LRMultiplier('adam', {'Conv': 0.5, 'Output': 1.5}),
    model.compile(loss='categorical_crossentropy', 
     optimizer=SGD(lr=0.01), 
        metrics=['accuracy'])
    model.summary()

    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*BATCH_SIZE)
    #callbacks = [lr_reducer, lr_scheduler,cp_callback]
    callbacks = [cp_callback]

    model.fit(train_imgs, train_labels,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(test_imgs, test_labels),
              shuffle=True,
              callbacks=callbacks)
    scores = model.evaluate(test_imgs, test_labels, verbose=1)
    
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save("testnet.h5")

if __name__ == '__main__':
    ## be nice to others
    tensorflow.config.threading.set_inter_op_parallelism_threads(8)
    tensorflow.config.threading.set_intra_op_parallelism_threads(8)
    epochs = 10 
    layer_num = 4
    train(epochs,layer_num)
