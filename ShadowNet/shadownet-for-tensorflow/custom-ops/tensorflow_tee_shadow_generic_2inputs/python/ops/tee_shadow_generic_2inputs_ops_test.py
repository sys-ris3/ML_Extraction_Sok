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
"""Tests for tee_shadow_generic ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_tee_shadow_generic_2inputs.python.ops.tee_shadow_generic_2inputs_ops import tee_shadow_generic_2inputs
except ImportError:
  from tee_shadow_generic_2inputs_ops import tee_shadow_generic_2inputs


class TeeShadowGenericTest(test.TestCase):

  def testTeeShadowGeneric(self):
    with self.test_session():
      self.assertAllClose(
          tee_shadow_generic_2inputs(pos="conv1", h=2,w=2,c=2, input1=[[[[1.0, 2.0, 3.0], 
                 [3.0, 4.0, 5.0]], 
                [[4.0, 3.0, 2.0],
                 [2.0, 1.0, 1.0]]]],input2= [[[[1.0, 2.0, 3.0], 
                 [3.0, 4.0, 5.0]], 
                [[4.0, 3.0, 2.0],
                 [2.0, 1.0, 1.0]]]]),
                np.array(
                [[[[3.2, 3.1], 
                 [5.4, 5.3]], 
                [[2.3, 2.4],
                 [1.1, 1.2]]]]))


if __name__ == '__main__':
  test.main()
