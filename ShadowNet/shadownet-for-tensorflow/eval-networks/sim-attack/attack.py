import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine import training
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from model_converter import normal_conv_kernel_obf

import numpy as np
import random as rand

(train_imgs, train_labels), (test_imgs, test_labels) = cifar10.load_data()

def attack(input_shape,do_attack, victim_model, layer_num):
    if do_attack is True:
        obf_weights = []
        orig_model = tf.keras.models.load_model(victim_model) 
        for l in range(layer_num):
            original_conv_layer = orig_model.layers[l*2]
            assert('conv2d' in original_conv_layer.name)
            weight = original_conv_layer.get_weights()[0]
            #bias = original_conv_layer.get_weights()[1]
            arr, obf_dict, obf_weight = normal_conv_kernel_obf(weight, 1.2)
            obf_weights.append(obf_weight)
        model = attack_net(input_shape, obf_weights, True, layer_num)
    else:
        model = attack_net(input_shape, None, False,layer_num)
    return model

def attack_net(input_shape, conv_weights, reuse_weights, layer_num, num_classes=10):
    model = keras.Sequential()
    print("input_shape")
    print(input_shape)
    model.add(keras.Input(shape=input_shape))
    for i in range(layer_num):
        model.add(layers.Conv2D(76, 3, 1, use_bias=False))
        model.add(layers.Conv2D(64, 1, 1, activation="relu"))
        model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=8))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes,
        activation='softmax',
        kernel_initializer='he_normal'))
    print("layer name:")
    print(model.layers[0].name)
    if reuse_weights is True:
        for i in range(layer_num):
            model.layers[i*3].set_weights([conv_weights[i]])
            model.layers[i*3].trainable = False
    else:
        for i in range(layer_num):
            model.layers[i*3].trainable = False

    return model

def train(do_attack, epochs, layer_num):
    global train_imgs,train_labels,test_imgs,test_labels 
    # Training parameters
    BATCH_SIZE = 128  
    EPOCHS = epochs # 200
    USE_AUGMENTATION = True
    NUM_CLASSES = np.unique(train_labels).shape[0] # 10
    COLORS = train_imgs.shape[3]
    input_shape = train_imgs.shape[1:]
    SUBTRACT_PIXEL_MEAN = True
    checkpoint_path = "training_attack/cp-{epoch:04d}.ckpt"
    VERSION = 1


    # Normalize data.
    train_imgs = train_imgs.astype('float32') / 255
    test_imgs = test_imgs.astype('float32') / 255
    train_labels = tensorflow.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = tensorflow.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    if do_attack is True:
        print("Do attack by reuse weights!")
        model = attack(input_shape, do_attack, 'testnet.h5', layer_num)
    else:
        print("Just train, no attack!")
        model = attack(input_shape, do_attack, 'testnet.h5', layer_num)


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

    model.save("attack_net.h5")

if __name__ == '__main__':
    ## be nice to others
    tensorflow.config.threading.set_inter_op_parallelism_threads(8)
    tensorflow.config.threading.set_intra_op_parallelism_threads(8)
    epochs = 10
    layer_num = 4
    #train(True, epochs, layer_num)
    train(False, epochs, layer_num)
    #train(False)
    
