#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_linear_transform import linear_transform 
from tensorflow import keras
from tensorflow.keras import layers 

class LinearTransform(layers.Layer):
  def __init__(self, units, **kwargs):
    super(LinearTransform, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.m = self.add_weight(name='m',
                             shape=(2, self.units),
                             dtype='int32',
                             trainable=True)
    self.s = self.add_weight(name='s',
                             shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)
    super(LinearTransform, self).build(input_shape) 

  def call(self, inputs):
    return linear_transform(inputs, self.m, self.s)
  def get_config(self):
      base_config = super(LinearTransform, self).get_config()
      base_config['units'] = self.units
      return base_config

if __name__ == '__main__':
    print("LinearTransform: to be tested.")
