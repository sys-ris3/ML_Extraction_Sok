#!/usr/bin/env python
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import numpy as np

from tensorflow.lite.python import tflite_convert
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.training_util import write_graph

from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from linear_transform_generic_layer import LinearTransformGeneric 
from shuffle_channel_layer import ShuffleChannel 
from tee_shadow_generic_layer import TeeShadowGeneric 
from tee_shadow_layer import TeeShadow
import tensorflow as tf


#MODEL="mobilenet_obf_filled_plan_A.h5"
#MODEL="mobilenet_obf_custom_filled.h5"
#TFMODEL="mobilenet_obf_filled.tflite"
#MODEL="mobilenet_obf_custom.h5"
#TFMODEL="mobilenet_obf.tflite"
#MODEL="mobilenet_obf_bhalf_filled.h5"
#TFMODEL="mobilenet_obf_bhalf.tflite"
#MODEL="alexnetobf.h5"
#TFMODEL="alexnetobf.tflite"
#MODEL="minivggobf.h5"
#TFMODEL="minivggobf.tflite"
#MODEL="inception_v3_obf.h5"
#TFMODEL="inception_v3_obf.tflite"
MODEL="minivggsplit.h5"
TFMODEL="minivggsplit.tflite"

class TestConvert():
  def _getKerasModelFile(self):
    model = tf.keras.models.load_model(MODEL, custom_objects={'TeeShadow':TeeShadow,'TeeShadowGeneric':TeeShadowGeneric,'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel,'LinearTransformGeneric':LinearTransformGeneric})
    model.summary()
    return model

  def _run(self, flags_str, input_file, output):
    output_file = os.path.join('.', output)
    #tflite_bin = resource_loader.get_path_to_datafile('tflite_convert')
    tflite_bin = "tflite_convert"
    cmdline = '{0} --output_file={1} {2}'.format(tflite_bin, output_file,
                                                 flags_str)

    exitcode = os.system(cmdline)
    if exitcode == 0:
      with gfile.Open(output_file, 'rb') as model_file:
        content = model_file.read()
      #os.remove(output_file)
    else:
      print("run cmd failed! cmd:"+cmdline)

class TfLiteConvertV1Test(TestConvert):

  def _run(self, flags_str, input_file, output_file):
    if tf2.enabled():
      flags_str += ' --enable_v1_converter'
    super(TfLiteConvertV1Test, self)._run(flags_str, input_file, output_file)

  def testMobilenet(self, input_file, output_file):
    custom_opdefs_str = (
            'name: \'AddMask\' '
            'input_arg: { name: \'input\' type: DT_FLOAT } '
            'input_arg: { name: \'weights\' type: DT_FLOAT } '
            'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
            'output_arg: { name: \'masked\' type: DT_FLOAT } ')
#            'name: \'LinearTransform\' '
#            'input_arg: { name: \'input\' type: DT_FLOAT } '
#            'input_arg: { name: \'weights\' type: DT_INT32} '
#            'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
#            'output_arg: { name: \'transformed\' type: DT_FLOAT } '
#            'name: \'ShuffleChannel\' '
#            'input_arg: { name: \'input\' type: DT_FLOAT } '
#            'input_arg: { name: \'weights\' type: DT_INT32} '
#            'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
#            'output_arg: { name: \'shuffled\' type: DT_FLOAT } ')
#    custom_opdefs_str = (
#        'name: \'TFLite_Detection_PostProcess\' '
#        'input_arg: { name: \'raw_outputs/box_encodings\' type: DT_FLOAT } '
#        'input_arg: { name: \'raw_outputs/class_predictions\' type: DT_FLOAT } '
#        'input_arg: { name: \'anchors\' type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess\' type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess:1\' '
#        'type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess:2\' '
#        'type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess:3\' '
#        'type: DT_FLOAT } '
#        'attr : { name: \'h_scale\' type: \'float\'} '
#        'attr : { name: \'max_classes_per_detection\' type: \'int\'} '
#        'attr : { name: \'max_detections\' type: \'int\'} '
#        'attr : { name: \'nms_iou_threshold\' type: \'float\'} '
#        'attr : { name: \'nms_score_threshold\' type: \'float\'} '
#        'attr : { name: \'num_classes\' type: \'int\'} '
#        'attr : { name: \'w_scale\' type: \'int\'} '
#        'attr : { name: \'x_scale\' type: \'int\'} '
#        'attr : { name: \'y_scale\' type: \'int\'}')

    keras_file = input_file 
    flags_str = ('--keras_model_file={0} --custom_opdefs="{1}" '
                 .format(keras_file, custom_opdefs_str))

    # Valid conversion.
    flags_str_final = ('{} --allow_custom_ops '
                       '--experimental_new_converter').format(flags_str)
    self._run(flags_str_final, input_file, output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='tflite-cmd-converter')
  parser.add_argument('-i', '--input-file', default = 'model.h5',
                    help = 'the path of %(prog)s input file')
  parser.add_argument('-o', '--output-file', default = 'model.tflite',
                    help = 'the path of %(prog)s output file')

  args = parser.parse_args()
  tfcv1 = TfLiteConvertV1Test()
  tfcv1.testMobilenet(args.input_file, args.output_file)
