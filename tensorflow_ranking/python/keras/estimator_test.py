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
"""Tests for Keras Estimator."""

import tensorflow as tf

from tensorflow_ranking.python.keras import estimator as estimator_lib
from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import metrics
from tensorflow_ranking.python.keras import model
from tensorflow_ranking.python.keras import network

_SIZE = "example_list_size"


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
              dense_shape=[2, 2, 1]),
      _SIZE:
          tf.convert_to_tensor(value=[[1], [2]])
  }


class _DummyUnivariateRankingNetwork(network.UnivariateRankingNetwork):
  """Dummy univariate ranking network with constant scoring function."""

  def score(self, context_features=None, example_features=None, training=True):
    large_batch_size = tf.shape(input=example_features["utility"])[0]
    return tf.ones(shape=(large_batch_size, 1))


class KerasModelToEstimatorTest(tf.test.TestCase):

  def setUp(self):
    super(KerasModelToEstimatorTest, self).setUp()
    self.context_feature_columns = _context_feature_columns()
    self.example_feature_columns = _example_feature_columns()
    self.features = _features()
    self.network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns)
    self.loss = losses.get(
        losses.RankingLossKey.SOFTMAX_LOSS,
        reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.eval_metrics = metrics.default_keras_metrics()
    self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    self.config = tf.estimator.RunConfig(
        keep_checkpoint_max=2, save_checkpoints_secs=2)

  def test_model_to_estimator(self):
    keras_model = model.create_keras_model(
        network=self.network,
        loss=self.loss,
        metrics=self.eval_metrics,
        optimizer=self.optimizer,
        size_feature_name=_SIZE)
    estimator = estimator_lib.model_to_estimator(
        model=keras_model, config=self.config)
    self.assertIsInstance(estimator, tf.compat.v1.estimator.Estimator)


if __name__ == "__main__":
  tf.test.main()
