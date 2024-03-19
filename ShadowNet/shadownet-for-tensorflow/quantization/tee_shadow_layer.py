#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import random as rand
from tensorflow_tee_shadow import tee_shadow 
from tensorflow import keras
from tensorflow.keras import layers 

class TeeShadow(layers.Layer):
  def __init__(self, units=0, position="conv1",  **kwargs):
    super(TeeShadow, self).__init__(**kwargs)
    self.units = units 
    self.position = position 

  def call(self, inputs):
    return tee_shadow(units = self.units, pos = self.position, input = inputs)
  def get_config(self):
      base_config = super(TeeShadow, self).get_config()
      base_config['units'] = self.units
      base_config['position'] = self.position
      return base_config

if __name__ == '__main__':
    print ("TeeShadow Layer Test: TBD")
