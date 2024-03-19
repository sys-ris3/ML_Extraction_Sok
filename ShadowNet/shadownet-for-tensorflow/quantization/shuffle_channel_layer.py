#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_shuffle_channel import shuffle_channel 
from tensorflow import keras
from tensorflow.keras import layers 

class ShuffleChannel(layers.Layer):
  def __init__(self, **kwargs):
    super(ShuffleChannel, self).__init__(**kwargs)

  def build(self, input_shape):
    self.shuffle_array = self.add_weight(name='shuffle_array',
                             shape=(input_shape[-1],),
                             dtype='int32',
                             trainable=True)
    self.scalar = self.add_weight(name='scalar',
                             shape=(input_shape[-1],),
                             initializer='random_normal',
                             trainable=True)
    super(ShuffleChannel, self).build(input_shape)

  def call(self, inputs):
    return shuffle_channel(inputs, self.shuffle_array, self.scalar)
  def get_config(self):
      base_config = super(ShuffleChannel, self).get_config()
      return base_config

if __name__ == '__main__':
    print("ShuffleChannel: to be tested.")
