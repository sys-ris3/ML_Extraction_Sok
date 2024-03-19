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
"""Tests for add_mask ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_add_mask.python.ops.add_mask_ops import add_mask
except ImportError:
  from add_mask_ops import add_mask


class AddMaskTest(test.TestCase):

  def testAddMask(self):
    with self.test_session():
      self.assertAllClose(
          add_mask([[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0],[2.0, 1.0]], 0.2), np.array([[1.8, 2.6], [3.4, 4.2]]))


if __name__ == '__main__':
  test.main()
