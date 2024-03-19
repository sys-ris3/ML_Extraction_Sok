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
"""Tests for shuffle_channel ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_shuffle_channel.python.ops.shuffle_channel_ops import shuffle_channel
except ImportError:
  from shuffle_channel_ops import shuffle_channel


class ShuffleChannelTest(test.TestCase):

  def testShuffleChannel(self):
    with self.test_session():
      self.assertAllClose(
          shuffle_channel(
              [[[[1., 2., 3.], 
                 [3., 4., 5.]], 
                [[4., 3., 2.],
                 [2., 1., 1.]]]],
                [2, 0, 1],
                [.1, .1, .1]),
                np.array(
              [[[[.3, .1, .2], 
                 [.5, .3, .4]], 
                [[.2, .4, .3],
                 [.1, .2, .1]]]]))

if __name__ == '__main__':
  test.main()
