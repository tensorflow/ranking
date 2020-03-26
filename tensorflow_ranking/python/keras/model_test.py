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
"""Tests for Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import metrics
from tensorflow_ranking.python.keras import model as model_lib
from tensorflow_ranking.python.keras import network as network_lib


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


class _DummyUnivariateRankingNetwork(network_lib.UnivariateRankingNetwork):
  """Dummy univariate ranking network with constant scoring function."""

  def score(self, context_features=None, example_features=None, training=True):
    large_batch_size = tf.shape(example_features["utility"])[0]
    return tf.ones(shape=(large_batch_size, 1))


class FunctionalRankingModelTest(tf.test.TestCase):

  def setUp(self):
    super(FunctionalRankingModelTest, self).setUp()
    self.context_feature_columns = _context_feature_columns()
    self.example_feature_columns = _example_feature_columns()
    self.optimizer = tf.keras.optimizers.Adagrad()
    self.loss = losses.SoftmaxLoss()
    self.metrics = [metrics.NDCGMetric("ndcg_5", topn=5)]

  def test_create_keras_model_without_padding(self):
    network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns)
    ranker = model_lib.create_keras_model(
        network=network,
        loss=self.loss,
        metrics=self.metrics,
        optimizer=self.optimizer,
        size_feature_name=None)
    self.assertEqual(ranker.optimizer, self.optimizer)
    self.assertEqual(ranker.loss, self.loss)
    self.assertNotIn("example_list_size", ranker.input_names)

  def test_create_keras_model_with_padding(self):
    network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns)
    ranker = model_lib.create_keras_model(
        network=network,
        loss=self.loss,
        metrics=self.metrics,
        optimizer=self.optimizer,
        size_feature_name="example_list_size")
    self.assertEqual(ranker.optimizer, self.optimizer)
    self.assertEqual(ranker.loss, self.loss)
    self.assertIn("example_list_size", ranker.input_names)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
