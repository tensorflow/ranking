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
"""Test for Keras feature transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import feature


def _get_feature_columns():

  def _normalizer_fn(t):
    return tf.math.log1p(t * tf.sign(t)) * tf.sign(t)

  context_feature_columns = {
      'query_length':
          tf.feature_column.numeric_column(
              'query_length', shape=(1,), default_value=0, dtype=tf.int64)
  }
  example_feature_columns = {
      'utility':
          tf.feature_column.numeric_column(
              'utility',
              shape=(1,),
              default_value=0.0,
              dtype=tf.float32,
              normalizer_fn=_normalizer_fn),
      'unigrams':
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_vocabulary_list(
                  'unigrams',
                  vocabulary_list=[
                      'ranking', 'regression', 'classification', 'ordinal'
                  ]),
              dimension=10)
  }
  custom_objects = {'_normalizer_fn': _normalizer_fn}
  return context_feature_columns, example_feature_columns, custom_objects


def _features():
  return {
      'query_length':
          tf.convert_to_tensor(value=[[1], [2]]),
      'utility':
          tf.convert_to_tensor(value=[[[1.0], [0.0]], [[0.0], [1.0]]]),
      'unigrams':
          tf.SparseTensor(
              indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
              values=['ranking', 'regression', 'classification', 'ordinal'],
              dense_shape=[2, 2, 1]),
      'example_feature_size':
          tf.convert_to_tensor(value=[1, 2])
  }


def _clone_keras_obj(obj, custom_objects=None):
  return obj.__class__.from_config(
      obj.get_config(), custom_objects=custom_objects)


class KerasInputsTest(tf.test.TestCase):

  def setUp(self):
    super(KerasInputsTest, self).setUp()
    (context_feature_columns, example_feature_columns,
     _) = _get_feature_columns()
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns

  def test_keras_inputs_dynamic_list_shape(self):
    keras_inputs = feature.create_keras_inputs(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns,
        size_feature_name=None)

    self.assertEqual(keras_inputs['query_length'].shape.as_list(), [None, 1])
    self.assertEqual(keras_inputs['query_length'].dtype, tf.int64)

    self.assertEqual(keras_inputs['utility'].shape.as_list(), [None, None, 1])
    self.assertEqual(keras_inputs['utility'].dtype, tf.float32)

    self.assertEqual(keras_inputs['unigrams'].dtype, tf.string)


class EncodeListwiseFeaturesTest(tf.test.TestCase):

  def setUp(self):
    super(EncodeListwiseFeaturesTest, self).setUp()
    (context_feature_columns, example_feature_columns,
     custom_objects) = _get_feature_columns()
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    self._custom_objects = custom_objects

    # Batch size = 2, list_size = 2.
    self._features = _features()
    self._listwise_dense_layer = feature.EncodeListwiseFeatures(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns)

  def test_get_config(self):
    # Check save and restore config.
    restored_layer = _clone_keras_obj(
        self._listwise_dense_layer, custom_objects=self._custom_objects)
    self.assertEqual(restored_layer.context_feature_columns,
                     self._context_feature_columns)
    self.assertEqual(restored_layer.example_feature_columns['utility'],
                     self._example_feature_columns['utility'])
    # TODO: Deserialized embedding feature column behavior is the
    # same but config is different. Hence we check for individual attributes.
    self.assertEqual(restored_layer.example_feature_columns['unigrams'].name,
                     'unigrams_embedding')
    self.assertEqual(
        restored_layer.example_feature_columns['unigrams'].initializer.mean,
        0.0)
    self.assertCountEqual(
        restored_layer.example_feature_columns['unigrams'].categorical_column
        .vocabulary_list,
        ['ranking', 'regression', 'classification', 'ordinal'])

  def test_listwise_dense_layer(self):
    context_features, example_features = self._listwise_dense_layer(
        inputs=self._features, training=False)
    self.assertAllInSet(['query_length'], set(six.iterkeys(context_features)))
    self.assertAllInSet(['unigrams', 'utility'],
                        set(six.iterkeys(example_features)))
    self.assertAllEqual(example_features['unigrams'].get_shape().as_list(),
                        [2, 2, 10])
    self.assertAllEqual(context_features['query_length'], [[1], [2]])
    self.assertAllEqual(
        example_features['utility'],
        [[[tf.math.log1p(1.0)], [0.0]], [[0.0], [tf.math.log1p(1.0)]]])

  def test_create_keras_inputs(self):
    keras_inputs = feature.create_keras_inputs(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns,
        size_feature_name='example_list_size')

    self.assertCountEqual(
        keras_inputs.keys(),
        list(self._context_feature_columns.keys()) +
        list(self._example_feature_columns.keys()) + ['example_list_size'])

  def test_create_keras_inputs_sparse_features(self):
    context_feature_columns = {
        'query':
            tf.feature_column.categorical_column_with_vocabulary_list(
                'query',
                vocabulary_list=[
                    'ranking', 'regression', 'classification', 'ordinal'
                ])
    }
    example_feature_columns = {
        'title':
            tf.feature_column.categorical_column_with_vocabulary_list(
                'title',
                vocabulary_list=[
                    'ranking', 'regression', 'classification', 'ordinal'
                ])
    }
    keras_inputs = feature.create_keras_inputs(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        size_feature_name='example_list_size')

    self.assertCountEqual(
        keras_inputs.keys(),
        list(context_feature_columns.keys()) +
        list(example_feature_columns.keys()) + ['example_list_size'])


class GenerateMaskTest(tf.test.TestCase):

  def setUp(self):
    super(GenerateMaskTest, self).setUp()
    (context_feature_columns, example_feature_columns,
     custom_objects) = _get_feature_columns()
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    self._custom_objects = custom_objects

    # Batch size = 2, list_size = 2.
    self._features = _features()
    self._mask_generator_layer = feature.GenerateMask(
        example_feature_columns=self._example_feature_columns,
        size_feature_name='example_feature_size')

  def test_get_config(self):
    # Check save and restore config.
    restored_layer = _clone_keras_obj(
        self._mask_generator_layer, custom_objects=self._custom_objects)
    self.assertEqual(restored_layer.example_feature_columns['utility'],
                     self._example_feature_columns['utility'])
    # TODO: Deserialized embedding feature column behavior is the
    # same but config is different. Hence we check for individual attributes.
    self.assertEqual(restored_layer.example_feature_columns['unigrams'].name,
                     'unigrams_embedding')
    self.assertEqual(
        restored_layer.example_feature_columns['unigrams'].initializer.mean,
        0.0)
    self.assertCountEqual(
        restored_layer.example_feature_columns['unigrams'].categorical_column
        .vocabulary_list,
        ['ranking', 'regression', 'classification', 'ordinal'])

  def test_mask_generator_layer(self):
    mask = self._mask_generator_layer(inputs=self._features, training=False)
    expected_mask = [[True, False], [True, True]]
    self.assertAllEqual(expected_mask, mask)


class FeatureColumnSerializationTest(tf.test.TestCase):

  def setUp(self):
    super(FeatureColumnSerializationTest, self).setUp()
    (_, example_feature_columns, custom_objects) = _get_feature_columns()
    self._feature_columns = example_feature_columns
    self._custom_objects = custom_objects

  def test_deserialization(self):
    serialized_feature_columns = feature.serialize_feature_columns(
        self._feature_columns)
    restored_feature_columns = feature.deserialize_feature_columns(
        serialized_feature_columns, custom_objects=self._custom_objects)
    self.assertEqual(restored_feature_columns['utility'],
                     self._feature_columns['utility'])
    # TODO: Deserialized embedding feature column behavior is the
    # same but config is different. Hence we check for individual attributes.
    self.assertEqual(restored_feature_columns['unigrams'].name,
                     'unigrams_embedding')
    self.assertEqual(restored_feature_columns['unigrams'].initializer.mean, 0.0)
    self.assertCountEqual(
        restored_feature_columns['unigrams'].categorical_column.vocabulary_list,
        ['ranking', 'regression', 'classification', 'ordinal'])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
