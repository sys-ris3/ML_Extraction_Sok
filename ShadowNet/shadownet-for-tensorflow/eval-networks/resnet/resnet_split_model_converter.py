#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform
from shuffle_channel_layer import ShuffleChannel 
from IPython import embed
from collections import OrderedDict

from resnet import resnet_v1_split
import json


from tensorflow.keras.utils import plot_model

def main():
    json_obj = OrderedDict()
    model = resnet_v1_split(json_obj)
    with open("split_model.json","w") as wfile:
        json.dump(json_obj,wfile)
    plot_model(model,'resnet-44-split.png')
    model.save("resnet-44-split.h5")
if __name__ == '__main__':
    main()
