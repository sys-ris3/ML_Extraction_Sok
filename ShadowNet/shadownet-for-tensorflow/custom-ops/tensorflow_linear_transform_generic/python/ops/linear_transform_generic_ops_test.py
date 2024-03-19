# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for linear_transform_generic ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_linear_transform_generic.python.ops.linear_transform_generic_ops import linear_transform_generic
except ImportError:
  from linear_transform_generic_ops import linear_transform_generic


class LinearTransformGenericTest(test.TestCase):

  def testLinearTransformGeneric(self):
    with self.test_session():
      self.assertAllClose(
          linear_transform_generic(
              [[[[1.0, 2.0, 3.0], 
                 [3.0, 4.0, 5.0]], 
                [[4.0, 3.0, 2.0],
                 [2.0, 1.0, 1.0]]]],
                [[1, 0],
                 [2, 2]],
                [0.1, 0.1]),
                [0, 0]),
                np.array(
                [[[[3.2, 3.1], 
                 [5.4, 5.3]], 
                [[2.3, 2.4],
                 [1.1, 1.2]]]]))


if __name__ == '__main__':
  test.main()
