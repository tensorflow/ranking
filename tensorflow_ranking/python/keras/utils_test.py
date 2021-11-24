# Copyright 2022 The TensorFlow Ranking Authors.
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

"""Tests for utils.py."""

import tensorflow.compat.v2 as tf
from tensorflow_ranking.python.keras import utils


class UtilsTest(tf.test.TestCase):

  def test_functions_are_serializable(self):
    for fn in [
        utils.identity,
        utils.inverse,
        utils.pow_minus_1,
        utils.log2_inverse,
        utils.is_greater_equal_1,
        utils.symmetric_log1p,
    ]:
      self.assertIsNotNone(tf.keras.utils.serialize_keras_object(fn))

  def test_functions_are_callable(self):
    self.assertEqual(utils.identity(1.0), 1.0)
    self.assertEqual(utils.inverse(1.0), 1.0)
    self.assertEqual(utils.pow_minus_1(1.0), 1.0)
    self.assertEqual(utils.log2_inverse(1.0), 1.0)
    self.assertEqual(utils.is_greater_equal_1(1.0), True)
    self.assertAllClose(utils.symmetric_log1p(-1.0), -0.69314718056)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
