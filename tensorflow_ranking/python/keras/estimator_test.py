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

# Lint as: python3
"""Tests for Keras Estimator."""

import os
from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.python import data
from tensorflow_ranking.python.keras import estimator as estimator_lib
from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import metrics
from tensorflow_ranking.python.keras import model
from tensorflow_ranking.python.keras import network
from tensorflow_serving.apis import input_pb2

_SIZE = 'example_list_size'

_ELWC_PROTO = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "query_length"
          value { int64_list { value: 3 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: "tensorflow" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 0.0 } }
        }
        feature {
          key: "dense_feature"
          value { float_list { value: -0.5 value: 0.5 } }
        }
        feature {
          key: "doc_weight"
          value { float_list { value: 0.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: ["learning", "to", "rank"] } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "dense_feature"
          value { float_list { value: 0.5 value: 0.5 } }
        }
        feature {
          key: "doc_weight"
          value { float_list { value: 1.0 } }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

_LABEL_FEATURE = 'utility'
_PADDING_LABEL = -1.
_EXAMPLE_WEIGHT_FEATURE = 'doc_weight'


def _get_feature_columns():

  def _normalizer_fn(t):
    return 2 * t

  context_feature_columns = {
      'query_length':
          tf.feature_column.numeric_column(
              'query_length',
              shape=(1,),
              default_value=0,
              dtype=tf.int64,
              normalizer_fn=_normalizer_fn)
  }
  example_feature_columns = {
      'utility':
          tf.feature_column.numeric_column(
              'utility',
              shape=(1,),
              default_value=_PADDING_LABEL,
              dtype=tf.float32),
      'unigrams':
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_vocabulary_list(
                  'unigrams',
                  vocabulary_list=[
                      'ranking', 'regression', 'classification', 'ordinal'
                  ]),
              dimension=10),
      'dense_feature':
          tf.feature_column.numeric_column(
              'dense_feature',
              shape=(2,),
              default_value=0.0,
              dtype=tf.float32)
  }
  custom_objects = {'_normalizer_fn': _normalizer_fn}
  return context_feature_columns, example_feature_columns, custom_objects


def _get_example_weight_feature_column():
  return tf.feature_column.numeric_column(
      _EXAMPLE_WEIGHT_FEATURE, dtype=tf.float32, default_value=1.)


# This network needs actual layers, otherwise the estimator training fails.
class _DummyUnivariateRankingNetwork(network.UnivariateRankingNetwork):
  """Dummy univariate ranking network with a simple scoring function."""

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               name='dummy_ranking_network',
               **kwargs):
    super(_DummyUnivariateRankingNetwork, self).__init__(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        name=name,
        **kwargs)
    self._score_layer = tf.keras.layers.Dense(units=1)

  def score(self, context_features=None, example_features=None, training=True):
    example_input = [
        tf.keras.layers.Flatten()(example_features[name])
        for name in sorted(self.example_feature_columns)
    ]
    return self._score_layer(tf.concat(example_input, axis=1))


class KerasModelToEstimatorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(KerasModelToEstimatorTest, self).setUp()
    (context_feature_columns, example_feature_columns,
     custom_objects) = _get_feature_columns()
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    # Remove label feature from example feature column.
    del self._example_feature_columns[_LABEL_FEATURE]

    self._custom_objects = custom_objects
    self._network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns)
    self._loss = losses.get(
        losses.RankingLossKey.SOFTMAX_LOSS,
        reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)
    self._eval_metrics = metrics.default_keras_metrics()
    self._optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    self._config = tf.estimator.RunConfig(
        keep_checkpoint_max=2, save_checkpoints_secs=2)

    self._data_file = os.path.join(tf.compat.v1.test.get_temp_dir(),
                                   'test_elwc.tfrecord')
    serialized_elwc_list = [
        _ELWC_PROTO.SerializeToString(),
    ] * 20
    if tf.io.gfile.exists(self._data_file):
      tf.io.gfile.remove(self._data_file)
    with tf.io.TFRecordWriter(self._data_file) as writer:
      for serialized_elwc in serialized_elwc_list:
        writer.write(serialized_elwc)

  def tearDown(self):
    super(KerasModelToEstimatorTest, self).tearDown()
    if tf.io.gfile.exists(self._data_file):
      tf.io.gfile.remove(self._data_file)
    self._data_file = None

  def _make_input_fn(self, weights_feature_name=None):
    """Return an input function, serves weights defined in weights_feature_name.

    Args:
      weights_feature_name: (str) A string defines the weights feature in
        dataset. None if no weights is used.

    Returns:
      A function serves features and labels. Weights will be served in features.
    """

    def _input_fn():
      context_feature_columns, example_feature_columns, _ = (
          _get_feature_columns())
      context_feature_spec = tf.feature_column.make_parse_example_spec(
          list(context_feature_columns.values()))

      label_column = tf.feature_column.numeric_column(
          _LABEL_FEATURE, dtype=tf.float32, default_value=_PADDING_LABEL)
      weight_column = (
          _get_example_weight_feature_column()
          if weights_feature_name == _EXAMPLE_WEIGHT_FEATURE else None)
      example_fc_list = (
          list(example_feature_columns.values()) + [label_column] +
          ([weight_column] if weight_column else []))
      example_feature_spec = tf.feature_column.make_parse_example_spec(
          example_fc_list)

      dataset = data.build_ranking_dataset(
          file_pattern=self._data_file,
          data_format=data.ELWC,
          batch_size=10,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          list_size=2,
          reader=tf.data.TFRecordDataset,
          size_feature_name=_SIZE)
      features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
      label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
      return features, label

    return _input_fn

  def test_model_to_estimator_missing_custom_objects(self):
    keras_model = model.create_keras_model(
        network=self._network,
        loss=self._loss,
        metrics=self._eval_metrics,
        optimizer=self._optimizer,
        size_feature_name=_SIZE)
    estimator = estimator_lib.model_to_estimator(
        model=keras_model, config=self._config, custom_objects=None)
    self.assertIsInstance(estimator, tf.compat.v1.estimator.Estimator)

    # Train and export model.
    train_spec = tf.estimator.TrainSpec(
        input_fn=self._make_input_fn(), max_steps=1)
    eval_spec = tf.estimator.EvalSpec(
        name='eval', input_fn=self._make_input_fn(), steps=10)

    with self.assertRaises(AttributeError):
      tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  @parameterized.named_parameters(
      ('without_weights', None, 'predict'),
      ('with_example_weights', _EXAMPLE_WEIGHT_FEATURE, 'predict'),
      ('pointwise_inference', None, 'regress'))
  def test_model_to_estimator(self, weights_feature_name, serving_default):
    keras_model = model.create_keras_model(
        network=self._network,
        loss=self._loss,
        metrics=self._eval_metrics,
        optimizer=self._optimizer,
        size_feature_name=_SIZE)
    estimator = estimator_lib.model_to_estimator(
        model=keras_model,
        config=self._config,
        weights_feature_name=weights_feature_name,
        custom_objects=self._custom_objects,
        serving_default=serving_default)
    self.assertIsInstance(estimator, tf.compat.v1.estimator.Estimator)

    # Train and export model.
    train_spec = tf.estimator.TrainSpec(
        input_fn=self._make_input_fn(weights_feature_name), max_steps=1)
    eval_spec = tf.estimator.EvalSpec(
        name='eval',
        input_fn=self._make_input_fn(weights_feature_name),
        steps=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    context_feature_spec = tf.feature_column.make_parse_example_spec(
        self._context_feature_columns.values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        self._example_feature_columns.values())

    def _make_serving_input_fn(serving_default):
      if serving_default == 'predict':
        return data.build_ranking_serving_input_receiver_fn(
            data.ELWC,
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
            size_feature_name=_SIZE)
      else:
        def pointwise_serving_fn():
          serialized = tf.compat.v1.placeholder(
              dtype=tf.string, shape=[None], name='input_ranking_tensor')
          receiver_tensors = {'input_ranking_data': serialized}
          features = data.parse_from_tf_example(
              serialized,
              context_feature_spec=context_feature_spec,
              example_feature_spec=example_feature_spec,
              size_feature_name=_SIZE)
          return tf.estimator.export.ServingInputReceiver(features,
                                                          receiver_tensors)
        return pointwise_serving_fn

    serving_input_receiver_fn = _make_serving_input_fn(serving_default)
    export_dir = os.path.join(tf.compat.v1.test.get_temp_dir(), 'export')
    estimator.export_saved_model(export_dir, serving_input_receiver_fn)

    # Confirm model ran and created checkpoints and saved model.
    final_ckpt_path = os.path.join(estimator.model_dir, 'model.ckpt-1.meta')
    self.assertTrue(tf.io.gfile.exists(final_ckpt_path))

    saved_model_pb = os.path.join(export_dir,
                                  tf.io.gfile.listdir(export_dir)[0],
                                  'saved_model.pb')
    self.assertTrue(tf.io.gfile.exists(saved_model_pb))

  def test_model_to_estimator_wrong_weights_name(self):
    keras_model = model.create_keras_model(
        network=self._network,
        loss=self._loss,
        metrics=self._eval_metrics,
        optimizer=self._optimizer,
        size_feature_name=_SIZE)
    estimator = estimator_lib.model_to_estimator(
        model=keras_model,
        config=self._config,
        weights_feature_name='weights',
        custom_objects=self._custom_objects)
    self.assertIsInstance(estimator, tf.compat.v1.estimator.Estimator)

    # Train and export model.
    train_spec = tf.estimator.TrainSpec(
        input_fn=self._make_input_fn(), max_steps=1)
    eval_spec = tf.estimator.EvalSpec(
        name='eval', input_fn=self._make_input_fn(), steps=10)

    with self.assertRaises(ValueError):
      tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.test.main()
