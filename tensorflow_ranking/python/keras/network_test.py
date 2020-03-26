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
"""Tests for Keras Ranking Network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

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


class _DummyRankingNetwork(network_lib.RankingNetwork):
  """Dummy ranking network with compute_logits that returns a constant logits."""

  def compute_logits(self,
                     context_features=None,
                     example_features=None,
                     training=True,
                     mask=None):
    batch_size, list_size, _ = example_features["utility"].get_shape().as_list()
    return tf.ones(shape=(batch_size, list_size))


class _DummyUnivariateRankingNetwork(network_lib.UnivariateRankingNetwork):
  """Dummy univariate ranking network with constant scoring function."""

  def score(self, context_features=None, example_features=None, training=True):
    large_batch_size = tf.shape(example_features["utility"])[0]
    return tf.ones(shape=(large_batch_size, 1))


class BaseRankingNetworkTest(tf.test.TestCase):

  def setUp(self):
    super(BaseRankingNetworkTest, self).setUp()
    self.context_feature_columns = _context_feature_columns()
    self.example_feature_columns = _example_feature_columns()
    self.features = _features()
    self.ranker = _DummyRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns)

  def test_transform_fn(self):
    context_features, example_features = self.ranker.transform(
        features=self.features, training=False)
    self.assertAllEqual(["query_length"], sorted(context_features))
    self.assertAllEqual(["unigrams", "utility"], sorted(example_features))
    self.assertAllEqual([2, 2, 10],
                        example_features["unigrams"].get_shape().as_list())
    self.assertAllEqual([[1], [2]], context_features["query_length"])
    self.assertAllEqual([[[1.0], [0.0]], [[0.0], [1.0]]],
                        example_features["utility"])

  def compute_logits_abstract_method(self):
    context_features, example_features = self.ranker.transform(
        features=self.features, training=False)
    with self.assertRaisesRegexp(
        NotImplementedError, r"Calling an abstract method, "
        "tfr.keras.RankingNetwork.compute_logits()."):
      network_lib.RankingNetwork.compute_logits(
          context_features=context_features, example_features=example_features)

  def test_call(self):
    logits = self.ranker(inputs=self.features)
    self.assertAllEqual([[1, 1], [1, 1]], logits)


class UnivariateRankingNetworkTest(tf.test.TestCase):

  def setUp(self):
    super(UnivariateRankingNetworkTest, self).setUp()
    self.context_feature_columns = _context_feature_columns()
    self.example_feature_columns = _example_feature_columns()
    self.features = _features()
    self.ranker = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns)

  def test_call(self):
    logits = self.ranker(
        inputs=self.features, mask=[[True, False], [True, True]])
    self.assertAllEqual([2, 2], logits.get_shape().as_list())

  def test_call_no_mask(self):
    logits = self.ranker(inputs=self.features)
    self.assertAllEqual([2, 2], logits.get_shape().as_list())


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
