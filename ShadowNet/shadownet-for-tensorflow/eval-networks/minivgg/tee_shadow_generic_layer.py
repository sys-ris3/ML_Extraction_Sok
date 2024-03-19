#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import random as rand
from tensorflow_tee_shadow_generic import tee_shadow_generic 
from tensorflow import keras
from tensorflow.keras import layers 

class TeeShadowGeneric(layers.Layer):
  def __init__(self, h, w, c, position,  **kwargs):
    super(TeeShadowGeneric, self).__init__(**kwargs)
    self.h = h 
    self.w = w 
    self.c = c 
    self.position = position 

  def call(self, inputs):
    return tee_shadow_generic(h = self.h, w = self.w, c = self.c, pos = self.position, input = inputs)
  def get_config(self):
      base_config = super(TeeShadowGeneric, self).get_config()
      base_config['h'] = self.h
      base_config['w'] = self.w
      base_config['c'] = self.c
      base_config['position'] = self.position
      return base_config

if __name__ == '__main__':
    print ("TeeShadowGeneric Layer Test: TBD")
