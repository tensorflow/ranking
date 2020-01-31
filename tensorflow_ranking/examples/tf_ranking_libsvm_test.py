# Copyright 2020 The TensorFlow Ranking Authors.
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

"""Tests for tf_ranking_libsvm.py."""

import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf

from tensorflow_ranking.examples import tf_ranking_libsvm

FLAGS = flags.FLAGS

LIBSVM_DATA = """1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76
"""


class TfRankingLibSVMTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("single_head", None),
                                  ("multi_head", "softmax_loss"))
  def test_train_and_eval(self, secondary_loss):
    data_dir = tf.compat.v1.test.get_temp_dir()
    data_file = os.path.join(data_dir, "libvsvm.txt")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with open(data_file, "wt") as writer:
      writer.write(LIBSVM_DATA)

    output_dir = os.path.join(data_dir, secondary_loss or "")

    with flagsaver.flagsaver(
        train_path=data_file,
        vali_path=data_file,
        test_path=data_file,
        output_dir=output_dir,
        loss="pairwise_logistic_loss",
        secondary_loss=secondary_loss,
        num_train_steps=10,
        list_size=10,
        group_size=2,
        num_features=100):
      tf_ranking_libsvm.train_and_eval()

    if tf.io.gfile.exists(output_dir):
      tf.io.gfile.rmtree(output_dir)


if __name__ == "__main__":
  tf.test.main()
