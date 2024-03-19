#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_linear_transform_generic import linear_transform_generic 
from tensorflow import keras
from tensorflow.keras import layers 

class LinearTransformGeneric(layers.Layer):
  def __init__(self, units, **kwargs):
    super(LinearTransformGeneric, self).__init__(**kwargs)
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
    self.b = self.add_weight(name='b',
                             shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)
    super(LinearTransformGeneric, self).build(input_shape) 

  def call(self, inputs):
    return linear_transform_generic(inputs, self.m, self.s, self.b)
  def get_config(self):
      base_config = super(LinearTransformGeneric, self).get_config()
      base_config['units'] = self.units
      return base_config

if __name__ == '__main__':
    print("LinearTransformGeneric: to be tested.")
