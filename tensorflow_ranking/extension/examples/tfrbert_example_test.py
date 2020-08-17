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

"""Tests for tfrbert_example.py."""

import shutil
import tempfile

from absl import flags
from absl.testing import flagsaver

import tensorflow as tf

from tensorflow_ranking.extension.examples import tfrbert_example
from google.protobuf import text_format
from tensorflow_serving.apis import input_pb2

FLAGS = flags.FLAGS

_TESTDATA_DIR = "/org_tensorflow_ranking/tensorflow_ranking/extension/testdata/"


class BertRankingExampleTest(tf.test.TestCase):

  def setUp(self):
    super(BertRankingExampleTest, self).setUp()
    # Defines file paths for model directory, traing file and eval datasets.
    self._model_dir = tempfile.mkdtemp()
    self._train_file = self._model_dir + "/example.train"
    self._eval_file = self._model_dir + "/example.eval"
    self._write_train_eval_tfrecord_examples()

    # Bert config and checkpoint files.
    self._bert_base_dir = FLAGS.test_srcdir + _TESTDATA_DIR
    self._bert_init_ckpt = self._bert_base_dir + "bert_lite_tf2.ckpt"
    self._bert_config_file = self._bert_base_dir + "bert_lite_config.json"
    self._bert_max_seq_length = 4

  def tearDown(self):
    super(BertRankingExampleTest, self).tearDown()
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _write_train_eval_tfrecord_examples(self):
    elwc_example = text_format.Parse(
        """
        context: {}
        examples: {
          features: {
            feature: {
              key: "relevance"
              value: { int64_list: { value: [ 1 ] } }
            }
            feature: {
              key: "input_ids"
              value: { int64_list: { value: [ 1, 4, 3, 0 ] } }
            }
            feature: {
              key: "input_mask"
              value: { int64_list: { value: [ 1, 1, 1, 0 ] } }
            }
            feature: {
              key: "segment_ids"
              value: { int64_list: { value: [ 0, 0, 0, 1 ] } }
            }
          }
        }
        examples: {
          features: {
            feature: {
              key: "relevance"
              value: { int64_list: { value: [ 0 ] } }
            }
            feature: {
              key: "input_ids"
              value: { int64_list: { value: [ 2, 5, 8, 9 ] } }
            }
            feature: {
              key: "input_mask"
              value: { int64_list: { value: [ 1, 1, 1, 1 ] } }
            }
            feature: {
              key: "segment_ids"
              value: { int64_list: { value: [ 0, 0, 0, 0 ] } }
            }
          }
        }
      """, input_pb2.ExampleListWithContext())

    # Writes TFRecord examples for training.
    with tf.io.TFRecordWriter(self._train_file) as writer:
      for example in [elwc_example] * 10:
        writer.write(example.SerializeToString())

    # Writes TFRecord examples for evaluation.
    with tf.io.TFRecordWriter(self._eval_file) as writer:
      for example in [elwc_example] * 5:
        writer.write(example.SerializeToString())

  def test_experiment_run(self):
    # Stores all flags defined in bert_ranking.py.
    with flagsaver.flagsaver(
        train_input_pattern=self._train_file,
        eval_input_pattern=self._eval_file,
        learning_rate=0.001,
        train_batch_size=2,
        eval_batch_size=2,
        model_dir=self._model_dir,
        num_train_steps=5,
        num_eval_steps=2,
        loss="softmax_loss",
        local_training=True,
        list_size=5,
        dropout_rate=0.1,
        bert_config_file=self._bert_config_file,
        bert_init_ckpt=self._bert_init_ckpt,
        bert_max_seq_length=self._bert_max_seq_length,
        bert_num_warmup_steps=1):
      tfrbert_example.train_and_eval()


if __name__ == "__main__":
  tf.test.main()
