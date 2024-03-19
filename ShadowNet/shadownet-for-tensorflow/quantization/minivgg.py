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

def MiniVGG(input_shape, classes):
    img_input = layers.Input(shape=input_shape)
    x = Conv2D(32, (3,3), padding= 'same')(img_input)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding= 'same')(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3,3), padding= 'same')(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Activation("relu")(x)

    x = Conv2D(128, (3,3), padding= 'same')(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Activation("relu")(x)

    x = Conv2D(128, (3,3), padding= 'same')(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2))(x)
    x = Dropout(0.25)(x)

    #x = Flatten()(x)
    #x = Dense(512, activation='linear')(x)
    #x = BatchNormalization(axis = -1)(x) # Channel last
    #x = Activation("relu")(x)
    #x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Reshape((1,1,x.shape[1]))(x)
    x = Conv2D(512, (1,1), padding= 'same')(x)
    x = BatchNormalization(axis = -1)(x) # Channel last
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Reshape((x.shape[3],))(x)

    x = Dense(classes, activation= 'softmax')(x)

    model = training.Model(img_input, x, name='minivgg')
    return model

def plot_train_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('train_history.png')
    return

def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model = MiniVGG((32,32,3),10)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=10, 
                        validation_data=(x_test, y_test))
    plot_train_history(history)
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    model.save("trained_minivgg.h5", include_optimizer=False)

def compare(m_ori, m_obf, quantize=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    model_ori = tf.keras.models.load_model(m_ori)
    print(model_ori.summary())
    model_obf = tf.keras.models.load_model(m_obf, custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ActivationQ':ActivationQ})
    #printf("ori_loss:%f,ori_acc:%f, obf_loss:%f,obf_acc:%f" % (ori_loss, ori_acc, obf_loss, obf_acc))
    for i in range(10):
        test1 = np.reshape(x_test[i*10], (1,) + x_test[i*10].shape)
        test1q = test1.astype('float32')*256.0
        ori = model_ori.predict(test1)
        obf = model_obf.predict(test1q)
        print("for test[%d]:"%(i*10))
        print(ori)
        print(obf)
    

    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n\t%s train|compare"%sys.argv[0])
        exit(1)
    if sys.argv[1] == "train":
        is_train = True
    else:
        is_train = False 
    if is_train:
        train()
    else:
        m_ori = "trained_minivgg_77.h5"
        m_obfq = "trained_minivgg_77_auto_obf_quant.h5"
        #m_obfq = "minivgg_auto_obf_quant.h5"
        compare(m_ori, m_obfq, quantize=True)
