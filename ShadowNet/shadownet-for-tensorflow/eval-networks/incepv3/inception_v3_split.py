# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
"""Inception V3 model for Keras.

Reference paper:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export

from tee_shadow_layer import TeeShadow


WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

layer_id = 0

@keras_export('keras.applications.inception_v3.InceptionV3',
              'keras.applications.InceptionV3')
def InceptionV3Obf(
    obf_ratio = 1.2,
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
):
  """Instantiates the Inception v3 architecture.

  Reference paper:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in the `tf.keras.backend.image_data_format()`.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.inception_v3.preprocess_input` for an example.

  Arguments:
    include_top: Boolean, whether to include the fully-connected
      layer at the top, as the last layer of the network. Default to `True`.
    weights: One of `None` (random initialization),
      `imagenet` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Default to `imagenet`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model. `input_tensor` is useful for sharing
      inputs between multiple different networks. Default to None.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(299, 299, 3)` (with `channels_last` data format)
      or `(3, 299, 299)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 75.
      E.g. `(150, 150, 3)` would be one valid value.
      `input_shape` will be ignored if the `input_tensor` is provided.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` (default) means that the output of the model will be
          the 4D tensor output of the last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Default to 1000.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=299,
      min_size=75,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  if backend.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = 3

  x = conv2d_bn_obf(img_input, obf_ratio, 32, 3, 3, strides=(2, 2), padding='valid')
  x = conv2d_bn_obf(x, obf_ratio, 32, 3, 3, padding='valid')
  x = conv2d_bn_obf(x, obf_ratio, 64, 3, 3)

  x = conv2d_bn_obf(x, obf_ratio, 80, 1, 1, padding='valid')
  x = conv2d_bn_obf(x, obf_ratio, 192, 3, 3, padding='valid')

  # mixed 0: 35 x 35 x 256
  branch1x1 = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)


  branch5x5 = conv2d_bn_obf(x, obf_ratio, 48, 1, 1)
  branch5x5 = conv2d_bn_obf(branch5x5, obf_ratio, 64, 5, 5)

  branch3x3dbl = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 32, 1, 1)
  x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed0')

  # mixed 1: 35 x 35 x 288
  branch1x1 = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)

  branch5x5 = conv2d_bn_obf(x, obf_ratio, 48, 1, 1)
  branch5x5 = conv2d_bn_obf(branch5x5, obf_ratio, 64, 5, 5)

  branch3x3dbl = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 64, 1, 1)
  x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed1')

  # mixed 2: 35 x 35 x 288
  branch1x1 = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)

  branch5x5 = conv2d_bn_obf(x, obf_ratio, 48, 1, 1)
  branch5x5 = conv2d_bn_obf(branch5x5, obf_ratio, 64, 5, 5)

  branch3x3dbl = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 64, 1, 1)
  x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed2')

  # mixed 3: 17 x 17 x 768
  branch3x3 = conv2d_bn_obf(x, obf_ratio, 384, 3, 3, strides=(2, 2), padding='valid')

  branch3x3dbl = conv2d_bn_obf(x, obf_ratio, 64, 1, 1)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3)
  branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed3')

  # mixed 4: 17 x 17 x 768
  branch1x1 = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)

  branch7x7 = conv2d_bn_obf(x, obf_ratio, 128, 1, 1)
  branch7x7 = conv2d_bn_obf(branch7x7, obf_ratio, 128, 1, 7)
  branch7x7 = conv2d_bn_obf(branch7x7, obf_ratio, 192, 7, 1)

  branch7x7dbl = conv2d_bn_obf(x, obf_ratio, 128, 1, 1)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 128, 7, 1)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 128, 1, 7)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 128, 7, 1)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 192, 1, 7)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 192, 1, 1)
  x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed4')

  # mixed 5, 6: 17 x 17 x 768
  for i in range(2):
    branch1x1 = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)

    branch7x7 = conv2d_bn_obf(x, obf_ratio, 160, 1, 1)
    branch7x7 = conv2d_bn_obf(branch7x7, obf_ratio, 160, 1, 7)
    branch7x7 = conv2d_bn_obf(branch7x7, obf_ratio, 192, 7, 1)

    branch7x7dbl = conv2d_bn_obf(x, obf_ratio, 160, 1, 1)
    branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 160, 7, 1)
    branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 160, 1, 7)
    branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 160, 7, 1)
    branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(
                                              x)
    branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed' + str(5 + i))

  # mixed 7: 17 x 17 x 768
  branch1x1 = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)

  branch7x7 = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)
  branch7x7 = conv2d_bn_obf(branch7x7, obf_ratio, 192, 1, 7)
  branch7x7 = conv2d_bn_obf(branch7x7, obf_ratio, 192, 7, 1)

  branch7x7dbl = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 192, 7, 1)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 192, 1, 7)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 192, 7, 1)
  branch7x7dbl = conv2d_bn_obf(branch7x7dbl, obf_ratio, 192, 1, 7)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(x)
  branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 192, 1, 1)
  x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed7')

  # mixed 8: 8 x 8 x 1280
  branch3x3 = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)
  branch3x3 = conv2d_bn_obf(branch3x3, obf_ratio, 320, 3, 3, strides=(2, 2), padding='valid')

  branch7x7x3 = conv2d_bn_obf(x, obf_ratio, 192, 1, 1)
  branch7x7x3 = conv2d_bn_obf(branch7x7x3, obf_ratio, 192, 1, 7)
  branch7x7x3 = conv2d_bn_obf(branch7x7x3, obf_ratio, 192, 7, 1)
  branch7x7x3 = conv2d_bn_obf(branch7x7x3, obf_ratio, 192, 3, 3, strides=(2,2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                         axis=channel_axis,
                         name='mixed8')

  # mixed 9: 8 x 8 x 2048
  for i in range(2):
    branch1x1 = conv2d_bn_obf(x, obf_ratio, 320, 1, 1)

    branch3x3 = conv2d_bn_obf(x, obf_ratio, 384, 1, 1)
    branch3x3_1 = conv2d_bn_obf(branch3x3, obf_ratio, 384, 1, 3)
    branch3x3_2 = conv2d_bn_obf(branch3x3, obf_ratio, 384, 3, 1)
    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                   axis=channel_axis,
                                   name='mixed9_' + str(i))

    branch3x3dbl = conv2d_bn_obf(x, obf_ratio, 448, 1, 1)
    branch3x3dbl = conv2d_bn_obf(branch3x3dbl, obf_ratio, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn_obf(branch3x3dbl, obf_ratio, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn_obf(branch3x3dbl, obf_ratio, 384, 3, 1)
    branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                      axis=channel_axis)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(
                                              x)
    branch_pool = conv2d_bn_obf(branch_pool, obf_ratio, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed' + str(9 + i))
  if include_top:
    # Classification block
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    #x = layers.Dense(classes, activation=classifier_activation,
    #                 name='predictions')(x)
    x = layers.Reshape((1, 1, 2048))(x)
    x = conv2d_bn_obf(x, obf_ratio, scalar_stack, classes, 1, 1, ac=classifier_activation, use_bn=False, name='predictions')
    x = layers.Reshape((1000,))(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='inception_v3_obf')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      weights_path = data_utils.get_file(
          'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
    else:
      weights_path = data_utils.get_file(
          'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='bcbd6486424b2319ff4ef7d526e38f63')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def conv2d_bn_obf(x,
              obf_ratio,
              scalar_stack,
              filters,
              num_row,
              num_col,
              use_mask = True,
              use_bn = True,
              ac = 'relu',
              padding='same',
              strides=(1, 1),
              name=None):
  """Utility function to apply conv + BN.

  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    num_row: height of the convolution kernel.
    num_col: width of the convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'`
      for the convolution and `name + '_bn'` for the
      batch norm layer.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  #if name is not None:
  #  bn_name = name + '_bn_obf'
  #  conv_name = name + '_conv_obf'
  #else:
  #  bn_name = None
  #  conv_name = None
  global layer_id
  conv_name = 'conv2d_obf_%d'%layer_id
  bn_name = 'bn_%d'%layer_id
  if backend.image_data_format() == 'channels_first':
    bn_axis = 1
  else:
    bn_axis = 3

  if (use_mask is True):
    random_scalar = get_random_scalar(scalar_stack, True)
    x = AddMask(random_scalar, name = 'push_mask_%d'%layer_id)(x)

  obf_filters = int(filters*obf_ratio)
  x = layers.Conv2D(
      obf_filters, (num_row, num_col),
      strides=strides,
      padding=padding,
      use_bias=False,
      name=conv_name)(
          x)
  x = LinearTransform(filters, name='lt_%d'%layer_id)(x)

  if (use_mask is True):
    random_scalar = get_random_scalar(scalar_stack, False)
    x = AddMask(random_scalar, name = 'pop_mask_%d'%layer_id)(x)

  if use_bn is True:
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  x = layers.Activation(ac, name=name)(x)

  layer_id = layer_id + 1

  return x


@keras_export('keras.applications.inception_v3.preprocess_input')
def preprocess_input(x, data_format=None):
  """Preprocesses a numpy array encoding a batch of images.

  Arguments
    x: A 4D numpy array consists of RGB values within [0, 255].

  Returns
    Preprocessed array.

  Raises
    ValueError: In case of unknown `data_format` argument.
  """
  return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


@keras_export('keras.applications.inception_v3.decode_predictions')
def decode_predictions(preds, top=5):
  """Decodes the prediction result from the model.

  Arguments
    preds: Numpy tensor encoding a batch of predictions.
    top: Integer, how many top-guesses to return.

  Returns
    A list of lists of top class prediction tuples
    `(class_name, class_description, score)`.
    One list of tuples per sample in batch input.

  Raises
    ValueError: In case of invalid shape of the `preds` array (must be 2D).
  """
  return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='', ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
