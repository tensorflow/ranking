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


def _get_feature_columns():

  def _normalizer_fn(t):
    return tf.math.log1p(t * tf.sign(t)) * tf.sign(t)

  context_feature_columns = {
      "query_length":
          tf.feature_column.numeric_column(
              "query_length", shape=(1,), default_value=0, dtype=tf.int64)
  }
  example_feature_columns = {
      "utility":
          tf.feature_column.numeric_column(
              "utility",
              shape=(1,),
              default_value=0.0,
              dtype=tf.float32,
              normalizer_fn=_normalizer_fn),
      "unigrams":
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_vocabulary_list(
                  "unigrams",
                  vocabulary_list=[
                      "ranking", "regression", "classification", "ordinal"
                  ]),
              dimension=10)
  }
  custom_objects = {"_normalizer_fn": _normalizer_fn}
  return context_feature_columns, example_feature_columns, custom_objects


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
              dense_shape=[2, 2, 1]),
      "example_feature_size":
          tf.convert_to_tensor(value=[1, 2])
  }


def _clone_keras_obj(obj, custom_objects=None):
  return obj.__class__.from_config(
      obj.get_config(), custom_objects=custom_objects)


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
    (context_feature_columns, example_feature_columns,
     custom_objects) = _get_feature_columns()
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    self._custom_objects = custom_objects
    self._features = _features()
    self._ranker = _DummyRankingNetwork(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns)

  def test_transform_fn(self):
    context_features, example_features = self._ranker.transform(
        features=self._features, training=False)
    self.assertAllEqual(["query_length"], sorted(context_features))
    self.assertAllEqual(["unigrams", "utility"], sorted(example_features))
    self.assertAllEqual([2, 2, 10],
                        example_features["unigrams"].get_shape().as_list())
    self.assertAllEqual([[1], [2]], context_features["query_length"])
    self.assertAllEqual(
        [[[tf.math.log1p(1.0)], [0.0]], [[0.0], [tf.math.log1p(1.0)]]],
        example_features["utility"])

  def compute_logits_abstract_method(self):
    context_features, example_features = self._ranker.transform(
        features=self.features, training=False)
    with self.assertRaisesRegexp(
        NotImplementedError, r"Calling an abstract method, "
        "tfr.keras.RankingNetwork.compute_logits()."):
      network_lib.RankingNetwork.compute_logits(
          context_features=context_features, example_features=example_features)

  def test_call(self):
    logits = self._ranker(inputs=self._features)
    self.assertAllEqual([[1, 1], [1, 1]], logits)


class UnivariateRankingNetworkTest(tf.test.TestCase):

  def setUp(self):
    super(UnivariateRankingNetworkTest, self).setUp()
    (context_feature_columns, example_feature_columns,
     custom_objects) = _get_feature_columns()
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    self._custom_objects = custom_objects
    self._features = _features()
    self._ranker = _DummyUnivariateRankingNetwork(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns)

  def test_call(self):
    logits = self._ranker(
        inputs=self._features, mask=[[True, False], [True, True]])
    self.assertAllEqual([2, 2], logits.get_shape().as_list())
    self.assertAllEqual(logits.numpy(), [[1., 0.], [1., 1.]])

  def test_call_no_mask(self):
    logits = self._ranker(inputs=self._features)
    self.assertAllEqual([2, 2], logits.get_shape().as_list())
    self.assertAllEqual(logits.numpy(), [[1., 1.], [1., 1.]])

  def test_listwise_scoring(self):
    context_features, example_features = self._ranker._listwise_dense_layer(
        self._features, training=False)
    logits = network_lib.listwise_scoring(
        self._ranker.score,
        context_features=context_features,
        example_features=example_features,
        training=False,
        mask=[[True, False], [True, True]])
    self.assertAllEqual(logits.numpy(), [[[1.], [0.]], [[1.], [1.]]])


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
