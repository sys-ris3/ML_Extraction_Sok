#!/usr/bin/env python
import tensorflow as tf
from IPython  import embed
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Layer,Conv2D,Dense,InputLayer,AveragePooling2D,MaxPooling2D,Activation,Flatten,Reshape,Dropout,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,Lambda,DepthwiseConv2D,ReLU, Input
model = Sequential()


model.add(Dense(32, input_shape=(5,)))
model.add(Dense(1, activation="sigmoid"))
model.add(Lambda(lambda x: tf.round(x)))

#model.compile("adam", "mae") # or cross entropy

embed()
model.save("test_lambda_model.h5")
