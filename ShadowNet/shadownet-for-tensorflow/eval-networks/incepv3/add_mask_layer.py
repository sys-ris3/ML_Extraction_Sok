#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import random as rand
from tensorflow_add_mask import add_mask
from tensorflow import keras
from tensorflow.keras import layers 

class AddMask(layers.Layer):
  def __init__(self, random_scalar=None, **kwargs):
    super(AddMask, self).__init__(**kwargs)
    self.random_scalar = random_scalar

  def build(self, input_shape):
    self.m = self.add_weight(shape=(input_shape[1:]),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return add_mask(inputs, self.m, self.random_scalar)
    #scaled_m = tf.math.scalar_mul(
    #            self.random_scalar, self.m
    #            )
    #return inputs + scaled_m 
  def get_config(self):
      base_config = super(AddMask, self).get_config()
      base_config['random_scalar'] = self.random_scalar
      return base_config

if __name__ == '__main__':
    rscalar = 0.2
    a = [[1.0,2.0], [3.0,4.0]]
    m = [[4.0,3.0], [2.0,1.0]]
    o1 = add_mask(a,m,rscalar)
    mt = tf.convert_to_tensor(m)
    at = tf.convert_to_tensor(a)
    o2 = at + tf.math.scalar_mul(rscalar, mt)
    print ("o1:")
    print (o1)
    print ("o2:")
    print (o2)
