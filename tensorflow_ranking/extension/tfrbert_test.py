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

"""Tests for tfrbert.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf

from google.protobuf import text_format
import tensorflow_ranking as tfr
from tensorflow_ranking.extension import tfrbert
from tensorflow_serving.apis import input_pb2

FLAGS = flags.FLAGS

_TESTDATA_DIR = "/org_tensorflow_ranking/tensorflow_ranking/extension/testdata/"


def _make_hparams(train_input_pattern,
                  eval_input_pattern,
                  model_dir,
                  list_size=5,
                  num_train_steps=1,
                  num_eval_steps=1,
                  checkpoint_secs=1,
                  num_checkpoints=2,
                  bert_config_file=None,
                  bert_init_ckpt=None,
                  bert_max_seq_length=4,
                  bert_num_warmup_steps=1):
  return dict(
      train_input_pattern=train_input_pattern,
      eval_input_pattern=eval_input_pattern,
      learning_rate=0.01,
      train_batch_size=8,
      eval_batch_size=8,
      checkpoint_secs=checkpoint_secs,
      num_checkpoints=num_checkpoints,
      num_train_steps=num_train_steps,
      num_eval_steps=num_eval_steps,
      loss="softmax_loss",
      list_size=list_size,
      listwise_inference=True,
      convert_labels_to_binary=False,
      model_dir=model_dir,
      bert_config_file=bert_config_file,
      bert_init_ckpt=bert_init_ckpt,
      bert_max_seq_length=bert_max_seq_length,
      bert_num_warmup_steps=bert_num_warmup_steps)


class TfrBertUtilTest(tf.test.TestCase):

  def setUp(self):
    super(TfrBertUtilTest, self).setUp()
    self._base_dir = FLAGS.test_srcdir + _TESTDATA_DIR
    self._bert_init_ckpt = self._base_dir + "bert_lite_tf2.ckpt"
    self._bert_config_file = self._base_dir + "bert_lite_config.json"
    self._bert_max_seq_length = 4
    self._bert_vocab_file = self._base_dir + "bert_lite_vocab.txt"
    self._bert_do_lower_case = True
    tf.compat.v1.reset_default_graph()

  def _create_tfrbert_util(self):
    return tfrbert.TFRBertUtil(
        bert_config_file=self._bert_config_file,
        bert_init_ckpt=self._bert_init_ckpt,
        bert_max_seq_length=self._bert_max_seq_length)

  def _create_tfrbert_util_with_vocab(self):
    return tfrbert.TFRBertUtil(
        bert_config_file=self._bert_config_file,
        bert_init_ckpt=self._bert_init_ckpt,
        bert_max_seq_length=self._bert_max_seq_length,
        bert_vocab_file=self._bert_vocab_file,
        do_lower_case=self._bert_do_lower_case)

  def test_init_tokenizer(self):
    bert_helper = self._create_tfrbert_util()
    self.assertIsNone(bert_helper._tokenizer)

    bert_helper = self._create_tfrbert_util_with_vocab()
    self.assertIsNotNone(bert_helper._tokenizer)

  def test_convert_to_elwc_size_mismatch(self):
    query = "test"
    documents = ["This is test", "This is simple test", "my test"]
    label_name = "label"
    labels = [1, 0]

    with self.assertRaises(ValueError):
      # Raises an error as `documents` and `labels` have different sizes.
      bert_helper = self._create_tfrbert_util()
      _ = bert_helper.convert_to_elwc(
          context=query,
          examples=documents,
          labels=labels,
          label_name=label_name)

  def test_truncate_seq_pair(self):
    bert_helper = self._create_tfrbert_util_with_vocab()
    tokens_a = ["a", "b", "c", "d"]
    tokens_b = ["e", "f", "g", "h"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 6)
    self.assertAllEqual(tokens_a, ["a", "b", "c"])
    self.assertAllEqual(tokens_b, ["e", "f", "g"])

    tokens_a = ["a", "b", "c", "d"]
    tokens_b = ["e", "f", "g", "h"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 5)
    self.assertAllEqual(tokens_a, ["a", "b", "c"])
    self.assertAllEqual(tokens_b, ["e", "f"])

    tokens_a = ["a"]
    tokens_b = ["e", "f", "g", "h"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 3)
    self.assertAllEqual(tokens_a, ["a"])
    self.assertAllEqual(tokens_b, ["e", "f"])

    tokens_a = ["a", "b", "c", "d"]
    tokens_b = ["e"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 4)
    self.assertAllEqual(tokens_a, ["a", "b", "c"])
    self.assertAllEqual(tokens_b, ["e"])

    tokens_a = ["a", "b", "c", "d"]
    tokens_b = []
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 3)
    self.assertAllEqual(tokens_a, ["a", "b", "c"])
    self.assertAllEqual(tokens_b, [])

    tokens_a = []
    tokens_b = ["e", "f", "g", "h"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 3)
    self.assertAllEqual(tokens_a, [])
    self.assertAllEqual(tokens_b, ["e", "f", "g"])

    tokens_a = ["a", "b", "c", "d"]
    tokens_b = ["e", "f", "g", "h"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 9)
    self.assertAllEqual(tokens_a, ["a", "b", "c", "d"])
    self.assertAllEqual(tokens_b, ["e", "f", "g", "h"])

    tokens_a = ["a", "b", "c", "d"]
    tokens_b = ["e", "f", "g", "h"]
    bert_helper._truncate_seq_pair(tokens_a, tokens_b, 1)
    self.assertAllEqual(tokens_a, ["a"])
    self.assertAllEqual(tokens_b, [])

  def test_to_bert_ids(self):
    sent_a = "test"
    sent_b = "This is simple test"
    self._bert_max_seq_length = 8
    bert_helper = self._create_tfrbert_util_with_vocab()
    input_ids, input_mask, segment_ids = bert_helper._to_bert_ids(
        sent_a, sent_b)
    expected_input_ids = [7, 5, 8, 1, 2, 4, 5, 8]
    expected_input_mask = [1, 1, 1, 1, 1, 1, 1, 1]
    expected_segment_ids = [0, 0, 0, 1, 1, 1, 1, 1]
    self.assertAllEqual(input_ids, expected_input_ids)
    self.assertAllEqual(input_mask, expected_input_mask)
    self.assertAllEqual(segment_ids, expected_segment_ids)

    input_ids, input_mask, segment_ids = bert_helper._to_bert_ids(sent_b)
    expected_input_ids = [7, 1, 2, 4, 5, 8, 0, 0]
    expected_input_mask = [1, 1, 1, 1, 1, 1, 0, 0]
    expected_segment_ids = [0, 0, 0, 0, 0, 0, 0, 0]
    self.assertAllEqual(input_ids, expected_input_ids)
    self.assertAllEqual(input_mask, expected_input_mask)
    self.assertAllEqual(segment_ids, expected_segment_ids)

  def test_convert_to_elwc(self):
    query = "test"
    documents = ["This", "This is simple test", "test"]
    label_name = "label"
    labels = [1, 0, 1]

    self._bert_max_seq_length = 8
    bert_helper = self._create_tfrbert_util_with_vocab()
    elwc = bert_helper.convert_to_elwc(
        context=query, examples=documents, labels=labels, label_name=label_name)

    expected_elwc = text_format.Parse(
        """
        examples: {
          features: {
            feature: {
              key: "label"
              value: { int64_list: { value: [ 1 ] } }
            }
            feature: {
              key: "input_ids"
              value: { int64_list: { value: [ 7, 5, 8, 1, 8, 0, 0, 0 ] } }
            }
            feature: {
              key: "input_mask"
              value: { int64_list: { value: [ 1, 1, 1, 1, 1, 0, 0, 0 ] } }
            }
            feature: {
              key: "segment_ids"
              value: { int64_list: { value: [ 0, 0, 0, 1, 1, 0, 0, 0 ] } }
            }
          }
        }
        examples: {
          features: {
            feature: {
              key: "label"
              value: { int64_list: { value: [ 0 ] } }
            }
            feature: {
              key: "input_ids"
              value: { int64_list: { value: [ 7, 5, 8, 1, 2, 4, 5, 8 ] } }
            }
            feature: {
              key: "input_mask"
              value: { int64_list: { value: [ 1, 1, 1, 1, 1, 1, 1, 1 ] } }
            }
            feature: {
              key: "segment_ids"
              value: { int64_list: { value: [ 0, 0, 0, 1, 1, 1, 1, 1 ] } }
            }
          }
        }
        examples: {
          features: {
            feature: {
              key: "label"
              value: { int64_list: { value: [ 1 ] } }
            }
            feature: {
              key: "input_ids"
              value: { int64_list: { value: [ 7, 5, 8, 5, 8, 0, 0, 0 ] } }
            }
            feature: {
              key: "input_mask"
              value: { int64_list: { value: [ 1, 1, 1, 1, 1, 0, 0, 0 ] } }
            }
            feature: {
              key: "segment_ids"
              value: { int64_list: { value: [ 0, 0, 0, 1, 1, 0, 0, 0 ] } }
            }
          }
        }""", input_pb2.ExampleListWithContext())

    self.assertEqual(
        text_format.MessageToString(expected_elwc),
        text_format.MessageToString(elwc))


class SimpleTFRBertClient(object):
  """A simple bert ranking pipeline created for the integration test."""

  def __init__(self, hparams):
    self._hparams = hparams
    self._util = tfrbert.TFRBertUtil(
        bert_config_file=self._hparams.get("bert_config_file"),
        bert_init_ckpt=self._hparams.get("bert_init_ckpt"),
        bert_max_seq_length=self._hparams.get("bert_max_seq_length"))

  def context_feature_columns(self):
    return {}

  def example_feature_columns(self):
    feature_columns = {
        "input_ids":
            tf.feature_column.numeric_column(
                "input_ids",
                shape=(self._hparams.get("bert_max_seq_length"),),
                default_value=0,
                dtype=tf.int64),
        "input_mask":
            tf.feature_column.numeric_column(
                "input_mask",
                shape=(self._hparams.get("bert_max_seq_length"),),
                default_value=0,
                dtype=tf.int64),
        "segment_ids":
            tf.feature_column.numeric_column(
                "segment_ids",
                shape=(self._hparams.get("bert_max_seq_length"),),
                default_value=0,
                dtype=tf.int64),
    }
    return feature_columns

  def get_estimator(self):
    util = tfrbert.TFRBertUtil(
        bert_config_file=self._hparams.get("bert_config_file"),
        bert_init_ckpt=self._hparams.get("bert_init_ckpt"),
        bert_max_seq_length=self._hparams.get("bert_max_seq_length"))

    network = tfrbert.TFRBertRankingNetwork(
        context_feature_columns=self.context_feature_columns(),
        example_feature_columns=self.example_feature_columns(),
        bert_config_file=self._hparams.get("bert_config_file"),
        bert_max_seq_length=self._hparams.get("bert_max_seq_length"),
        bert_output_dropout=0.1,
        name="tfrbert")

    loss = tfr.keras.losses.get(
        self._hparams.get("loss"),
        reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)

    metrics = tfr.keras.metrics.default_keras_metrics()

    config = tf.estimator.RunConfig(
        model_dir=self._hparams.get("model_dir"),
        keep_checkpoint_max=self._hparams.get("num_checkpoints"),
        save_checkpoints_secs=self._hparams.get("checkpoint_secs"))

    optimizer = util.create_optimizer(
        init_lr=self._hparams.get("learning_rate"),
        train_steps=self._hparams.get("num_train_steps"),
        warmup_steps=self._hparams.get("bert_num_warmup_steps"))

    ranker = tfr.keras.model.create_keras_model(
        network=network,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        size_feature_name="example_list_size")

    return tfr.keras.estimator.model_to_estimator(
        model=ranker,
        model_dir=self._hparams.get("model_dir"),
        config=config,
        warm_start_from=util.get_warm_start_settings(exclude="tfrbert"))

  def get_pipeline(self):
    return tfr.ext.pipeline.RankingPipeline(
        context_feature_columns=self.context_feature_columns(),
        example_feature_columns=self.example_feature_columns(),
        hparams=self._hparams,
        estimator=self.get_estimator(),
        label_feature_name="label",
        label_feature_type=tf.int64,
        size_feature_name="example_list_size")


class TFRBertRankingPipelineIntegrationTest(tf.test.TestCase):
  """BertRankingEstimator tests."""

  def setUp(self):
    super(TFRBertRankingPipelineIntegrationTest, self).setUp()
    tf.compat.v1.reset_default_graph()

    # Defines the Bert checkpoint and configurations.
    bert_dir = FLAGS.test_srcdir + _TESTDATA_DIR
    self._bert_init_ckpt = bert_dir + "bert_lite_tf2.ckpt"
    self._bert_config_file = bert_dir + "bert_lite_config.json"
    self._bert_max_seq_length = 4

    # Prepares model directory, and train and eval data.
    self._model_dir = tf.compat.v1.test.get_temp_dir()
    tf.io.gfile.makedirs(self._model_dir)
    self._data_file = os.path.join(self._model_dir, "elwc.tfrecord")
    self._write_train_eval_data(self._data_file)

  def tearDown(self):
    super(TFRBertRankingPipelineIntegrationTest, self).tearDown()
    if self._model_dir:
      tf.io.gfile.rmtree(self._model_dir)
    self._model_dir = None

  def _write_train_eval_data(self, data_file):
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
    with tf.io.TFRecordWriter(data_file) as writer:
      for example in [elwc_example] * 8:
        writer.write(example.SerializeToString())

  def test_pipeline(self):
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        num_train_steps=3,
        num_eval_steps=3,
        num_checkpoints=2,
        checkpoint_secs=1,
        model_dir=self._model_dir,
        bert_config_file=self._bert_config_file,
        bert_init_ckpt=self._bert_init_ckpt,
        bert_max_seq_length=self._bert_max_seq_length,
        bert_num_warmup_steps=1)
    pip = SimpleTFRBertClient(hparams).get_pipeline()
    pip.train_and_eval(local_training=True)

    required_patterns = [
        r"model\.ckpt\-\d\.data\-00000\-of\-00001",
        r"model\.ckpt\-\d\.index",
        r"model\.ckpt\-\d\.meta",
    ]
    output_files = tf.io.gfile.listdir(self._model_dir)
    for pattern in required_patterns:
      self.assertRegex(",".join(output_files), pattern)


if __name__ == "__main__":
  tf.test.main()
