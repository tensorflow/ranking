# Copyright 2021 The TensorFlow Ranking Authors.
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

"""Tests for strategy_utils.py."""

import os

import tensorflow as tf

from tensorflow_ranking.python.keras import strategy_utils


class StrategyUtilsTest(tf.test.TestCase):

  def test_get_strategy(self):
    self.assertIsNone(strategy_utils.get_strategy(None))

    self.assertIsInstance(
        strategy_utils.get_strategy("MirroredStrategy"),
        tf.distribute.MirroredStrategy)

    self.assertIsInstance(
        strategy_utils.get_strategy("MultiWorkerMirroredStrategy"),
        tf.distribute.MultiWorkerMirroredStrategy)

    with self.assertRaises(ValueError):
      strategy_utils.get_strategy("NotSupportedStrategy")

  def test_get_output_filepath(self):
    none_strategy = strategy_utils.get_strategy(None)
    self.assertEqual(
        strategy_utils.get_output_filepath("/tmp/test", none_strategy),
        "/tmp/test")

    mirrored_strategy = strategy_utils.get_strategy("MirroredStrategy")
    self.assertEqual(
        strategy_utils.get_output_filepath("/tmp/test", mirrored_strategy),
        "/tmp/test")

    mwms_strategy = strategy_utils.get_strategy("MultiWorkerMirroredStrategy")
    filepath = "/tmp/test"
    if mwms_strategy.cluster_resolver.task_type == "worker":
      filepath = os.path.join(
          filepath, "workertemp_" + str(mwms_strategy.cluster_resolver.task_id))
    self.assertEqual(
        strategy_utils.get_output_filepath("/tmp/test", mwms_strategy),
        filepath)


if __name__ == "__main__":
  tf.test.main()
