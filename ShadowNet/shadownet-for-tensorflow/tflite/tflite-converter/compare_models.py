#!/usr/bin/env python
from numpy import asarray
from numpy import save
from numpy import load

import numpy as np
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel 

ORIG_MODEL='testnet_obf_filled.h5'
model = tf.keras.models.load_model(ORIG_MODEL, custom_objects={'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})

# Test the TensorFlow model on random input data.
input_data = load('input_data.npy')
tf_results = model(tf.constant(input_data))

tflite_results = load('tflite_output.npy')

# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
    print ((tf_result.shape))
    print("a: %f, b:%f" % (tf_result[0][0][0], tflite_result[0][0][0]))
