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

# Lint as: python3
"""Tests for Premade DNNRankingNetwork."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import metrics
from tensorflow_ranking.python.keras import model
from tensorflow_ranking.python.keras.canned import dnn


def _context_feature_columns():
  return {
      "query_length":
          tf.feature_column.numeric_column(
              "query_length", shape=(1,), default_value=0, dtype=tf.int64)
  }


def _example_feature_columns():
  return {
      "utility":
          tf.feature_column.numeric_column(
              "utility", shape=(1,), default_value=0.0, dtype=tf.float32),
      "unigrams":
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_vocabulary_list(
                  "unigrams",
                  vocabulary_list=[
                      "ranking", "regression", "classification", "ordinal"
                  ]),
              dimension=10)
  }


def _features():
  return {
      "query_length":
          tf.convert_to_tensor(value=[[1], [2]]),
      "utility":
          tf.convert_to_tensor(value=[[[1.0], [0.0]], [[0.0], [1.0]]]),
      "unigrams":
          tf.SparseTensor(
              indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
              values=["ranking", "regression", "classification", "ordinal"],
              dense_shape=[2, 2, 1])
  }


class DNNRankingNetworkTest(tf.test.TestCase):

  def setUp(self):
    super(DNNRankingNetworkTest, self).setUp()
    self.context_feature_columns = _context_feature_columns()
    self.example_feature_columns = _example_feature_columns()
    self.features = _features()
    self.network = dnn.DNNRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns,
        hidden_layer_dims=["10", "10", "10"],
        activation=tf.nn.relu,
        dropout=0.5)

  def test_get_config(self):
    # Check save and restore config.
    restored_network = dnn.DNNRankingNetwork.from_config(
        self.network.get_config())
    self.assertEqual(restored_network.context_feature_columns,
                     self.context_feature_columns)
    self.assertEqual(restored_network.example_feature_columns["utility"],
                     self.network.example_feature_columns["utility"])
    # TODO: Deserialized embedding feature column behavior is the
    # same but config is different. Hence we check for individual attributes.
    self.assertEqual(restored_network.example_feature_columns["unigrams"].name,
                     "unigrams_embedding")
    self.assertEqual(
        restored_network.example_feature_columns["unigrams"].initializer.mean,
        0.0)
    self.assertCountEqual(
        restored_network.example_feature_columns["unigrams"].categorical_column
        .vocabulary_list,
        ["ranking", "regression", "classification", "ordinal"])

    self.assertEqual(restored_network._hidden_layer_dims, [10, 10, 10])
    self.assertEqual(restored_network._activation, tf.nn.relu)
    self.assertEqual(restored_network._dropout, 0.5)

  def test_call(self):
    logits = self.network(
        inputs=self.features, mask=[[True, False], [True, True]])
    self.assertAllEqual([2, 2], logits.get_shape().as_list())

  def test_call_no_mask(self):
    logits = self.network(inputs=self.features)
    self.assertAllEqual([2, 2], logits.get_shape().as_list())

  def test_call_none_context_feature_columns(self):
    network = dnn.DNNRankingNetwork(
        context_feature_columns=None,
        example_feature_columns=self.example_feature_columns,
        hidden_layer_dims=["10", "10"],
        activation=tf.nn.relu,
        dropout=0.5,
        name="dnn_ranking_network")
    logits = network(inputs=self.features, mask=[[True, False], [True, True]])
    self.assertAllEqual([2, 2], logits.get_shape().as_list())

  def test_call_none_example_feature_columns(self):
    with self.assertRaisesRegexp(
        ValueError, r"example_feature_columns or "
        "hidden_layer_dims must not be empty."):
      dnn.DNNRankingNetwork(
          context_feature_columns=self.context_feature_columns,
          example_feature_columns=None,
          hidden_layer_dims=["10", "10"],
          activation=tf.nn.relu,
          dropout=0.5,
          name="dnn_ranking_network")

  def test_call_empty_example_hidden_layer_dims(self):
    with self.assertRaisesRegexp(
        ValueError, r"example_feature_columns or "
        "hidden_layer_dims must not be empty."):
      dnn.DNNRankingNetwork(
          context_feature_columns=self.context_feature_columns,
          example_feature_columns=self.example_feature_columns,
          hidden_layer_dims=[],
          activation=tf.nn.relu,
          dropout=0.5,
          name="dnn_ranking_network")

  def test_model_compile_keras(self):
    # Specify the training configuration (optimizer, loss, metrics).
    optimizer = tf.keras.optimizers.RMSprop()
    loss = losses.SoftmaxLoss()
    eval_metrics = [metrics.NDCGMetric("ndcg_5", topn=5)]
    ranker = model.create_keras_model(
        network=self.network,
        loss=loss,
        metrics=eval_metrics,
        optimizer=optimizer,
        size_feature_name=None)

    self.assertIs(ranker.optimizer, optimizer)
    self.assertIs(ranker.loss, loss)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
