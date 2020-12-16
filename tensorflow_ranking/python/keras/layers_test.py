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
"""Tests for Keras layers in TF-Ranking."""
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_ranking.python.keras import layers

_EPSILON = 1e-10


class ConcatFeaturesTest(tf.test.TestCase):

  def test_call_with_circular_padding(self):
    context_features = {
        'context_feature_1': tf.constant([[1], [0]], dtype=tf.float32)
    }
    example_features = {
        'example_feature_1':
            tf.constant([[[1], [0], [-1]], [[0], [1], [0]]], dtype=tf.float32)
    }
    mask = tf.constant([[True, True, False], [True, False, False]],
                       dtype=tf.bool)
    expected_concat_tensor = tf.constant(
        [[[1., 1.], [1., 0.], [1., 1.]], [[0., 0.], [0., 0.], [0., 0.]]],
        dtype=tf.float32)
    concat_tensor = layers.ConcatFeatures(circular_padding=True)(
        context_features=context_features,
        example_features=example_features,
        mask=mask)
    self.assertAllClose(expected_concat_tensor, concat_tensor)

  def test_call_without_circular_padding(self):
    context_features = {
        'context_feature_1': tf.constant([[1], [0]], dtype=tf.float32)
    }
    example_features = {
        'example_feature_1':
            tf.constant([[[1], [0], [-1]], [[0], [1], [0]]], dtype=tf.float32)
    }
    mask = tf.constant([[True, True, False], [True, False, False]],
                       dtype=tf.bool)
    expected_concat_tensor = tf.constant(
        [[[1., 1.], [1., 0.], [1., -1.]], [[0., 0.], [0., 1.], [0., 0.]]],
        dtype=tf.float32)
    concat_tensor = layers.ConcatFeatures(circular_padding=False)(
        context_features=context_features,
        example_features=example_features,
        mask=mask)
    self.assertAllClose(expected_concat_tensor, concat_tensor)

  def test_serialization(self):
    layer = layers.ConcatFeatures()
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())


class FlattenListTest(tf.test.TestCase):

  def test_call_with_circular_padding(self):
    context_features = {
        'context_feature_1': tf.constant([[1], [0]], dtype=tf.float32)
    }
    example_features = {
        'example_feature_1':
            tf.constant([[[1], [0], [-1]], [[0], [1], [0]]], dtype=tf.float32)
    }
    mask = tf.constant([[True, True, False], [True, False, False]],
                       dtype=tf.bool)
    target_context_features = {
        'context_feature_1':
            tf.constant([[1], [1], [1], [0], [0], [0]], dtype=tf.float32)
    }
    target_example_features = {
        'example_feature_1':
            tf.constant([[1], [0], [1], [0], [0], [0]], dtype=tf.float32)
    }
    (flattened_context_features,
     flattened_example_features) = layers.FlattenList()(
         context_features=context_features,
         example_features=example_features,
         mask=mask)
    self.assertAllClose(target_context_features, flattened_context_features)
    self.assertAllClose(target_example_features, flattened_example_features)

  def test_call_without_circular_padding(self):
    context_features = {
        'context_feature_1': tf.constant([[1], [0]], dtype=tf.float32)
    }
    example_features = {
        'example_feature_1':
            tf.constant([[[1], [0], [-1]], [[0], [1], [0]]], dtype=tf.float32)
    }
    mask = tf.constant([[True, True, False], [True, False, False]],
                       dtype=tf.bool)
    expected_context_features = {
        'context_feature_1':
            tf.constant([[1], [1], [1], [0], [0], [0]], dtype=tf.float32)
    }
    expected_example_features = {
        'example_feature_1':
            tf.constant([[1], [0], [-1], [0], [1], [0]], dtype=tf.float32)
    }
    (flattened_context_features,
     flattened_example_features) = layers.FlattenList(circular_padding=False)(
         context_features=context_features,
         example_features=example_features,
         mask=mask)
    self.assertAllClose(expected_context_features, flattened_context_features)
    self.assertAllClose(expected_example_features, flattened_example_features)

  def test_call_raise_error(self):
    context_features = {
        'context_feature_1': tf.constant([[1], [0]], dtype=tf.float32)
    }
    example_features = {}
    mask = tf.constant([[True, True, False], [True, False, False]],
                       dtype=tf.bool)
    with self.assertRaises(ValueError):
      layers.FlattenList()(
          context_features=context_features,
          example_features=example_features,
          mask=mask)

  def test_serialization(self):
    layer = layers.FlattenList()
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())


class RestoreListTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.flattened_logits = tf.constant([1, 0.5, 2, 0, -1, 0], dtype=tf.float32)
    self.flattened_logits_2d = tf.constant([[1], [0.5], [2], [0], [-1], [0]],
                                           dtype=tf.float32)
    self.mask = tf.constant([[True, True, False], [True, False, False]],
                            dtype=tf.bool)

  def test_call(self):
    logits = layers.RestoreList()(
        flattened_logits=self.flattened_logits, mask=self.mask)
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllEqual([[1, 0.5, tf.math.log(_EPSILON)],
                         [0, tf.math.log(_EPSILON),
                          tf.math.log(_EPSILON)]], logits.numpy())
    logits = layers.RestoreList()(
        flattened_logits=self.flattened_logits_2d, mask=self.mask)
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllEqual([[1, 0.5, tf.math.log(_EPSILON)],
                         [0, tf.math.log(_EPSILON),
                          tf.math.log(_EPSILON)]], logits.numpy())

  def test_call_by_scatter(self):
    logits = layers.RestoreList(by_scatter=True)(
        flattened_logits=self.flattened_logits, mask=self.mask)
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllClose(
        [[1.5, 0.5, tf.math.log(_EPSILON)],
         [-1. / 3., tf.math.log(_EPSILON),
          tf.math.log(_EPSILON)]], logits.numpy())
    logits = layers.RestoreList(by_scatter=True)(
        flattened_logits=self.flattened_logits_2d, mask=self.mask)
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllClose(
        [[1.5, 0.5, tf.math.log(_EPSILON)],
         [-1. / 3., tf.math.log(_EPSILON),
          tf.math.log(_EPSILON)]], logits.numpy())

  def test_serialization(self):
    layer = layers.RestoreList(by_scatter=True)
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())

    logits = loaded(flattened_logits=self.flattened_logits, mask=self.mask)
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllClose(
        [[1.5, 0.5, tf.math.log(_EPSILON)],
         [-1. / 3., tf.math.log(_EPSILON),
          tf.math.log(_EPSILON)]], logits.numpy())
    logits = loaded(flattened_logits=self.flattened_logits_2d, mask=self.mask)
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllClose(
        [[1.5, 0.5, tf.math.log(_EPSILON)],
         [-1. / 3., tf.math.log(_EPSILON),
          tf.math.log(_EPSILON)]], logits.numpy())

  def test_call_raise_error(self):
    flattened_logits = tf.constant([1, 0.5, 2, 0, -1], dtype=tf.float32)
    flattened_logits_2d = tf.constant([[1, 0], [0.5, 0], [2, 0], [0, 0]],
                                      dtype=tf.float32)
    mask = tf.constant([[True, True, False], [True, False, False]],
                       dtype=tf.bool)
    with self.assertRaises(ValueError):
      layers.RestoreList()(flattened_logits=flattened_logits, mask=mask)
    with self.assertRaises(ValueError):
      layers.RestoreList()(flattened_logits=flattened_logits_2d, mask=mask)


class DocumentInteractionAttentionLayerTest(tf.test.TestCase,
                                            parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Batch size = 2, list_size = 3.
    self._inputs = tf.constant(
        [[[2., 1.], [2., 0.], [2., -1.]], [[1., 0.], [1., 1.], [1., 0.]]],
        dtype=tf.float32)
    self._mask = tf.constant([[True, True, False], [True, False, False]],
                             dtype=tf.bool)
    self._num_heads = 2
    self._head_size = 2
    self._num_layers = 2
    self._dropout_rate = 0.5

  def _get_din_layer(self, topk=None):
    return layers.DocumentInteractionAttention(
        num_heads=self._num_heads,
        head_size=self._head_size,
        num_layers=self._num_layers,
        topk=topk,
        dropout_rate=self._dropout_rate)

  @parameterized.named_parameters(('topk_none', None), ('topk', 1))
  def test_serialization(self, topk):
    # Check save and restore config.
    layer = self._get_din_layer(topk)
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())

  @parameterized.named_parameters(('topk_none', None), ('topk', 1))
  def test_deterministic_inference_behavior(self, topk):
    din_layer = self._get_din_layer(topk)
    output_1 = din_layer(inputs=self._inputs, training=False, mask=self._mask)
    output_2 = din_layer(inputs=self._inputs, training=False, mask=self._mask)
    self.assertAllClose(output_1, output_2)

  def test_call_topk_none(self):
    tf.random.set_seed(1)
    din_layer = self._get_din_layer(topk=None)
    output = din_layer(inputs=self._inputs, training=False, mask=self._mask)
    self.assertEqual(output.shape.as_list(), [2, 3, self._head_size])

    expected_output = tf.convert_to_tensor([[[-1., 1.0000001], [-1., 1.0000001],
                                             [-1., 1.0000001]],
                                            [[-1., 0.99999994], [-1., 1.],
                                             [-1., 1.]]])
    self.assertAllClose(expected_output, output)

  def test_call_topk(self):
    tf.random.set_seed(1)
    din_layer = self._get_din_layer(topk=1)
    output = din_layer(inputs=self._inputs, training=False, mask=self._mask)
    self.assertEqual(output.shape.as_list(), [2, 3, self._head_size])
    expected_output = tf.convert_to_tensor([[[-1., 1.], [-1., 1.], [-1., 1.]],
                                            [[-1., 0.99999994], [-1., 1.],
                                             [-1., 1.]]])
    self.assertAllClose(expected_output, output)

  @parameterized.named_parameters(('topk_none', None), ('topk', 1))
  def test_no_effect_circular_padding(self, topk):
    din_layer = self._get_din_layer(topk)
    output_1 = din_layer(inputs=self._inputs, training=False, mask=self._mask)
    circular_padded_inputs = tf.constant(
        [[[2., 1.], [2., 0.], [2., 1.]], [[1., 0.], [1., 0.], [1., 0.]]],
        dtype=tf.float32)
    output_2 = din_layer(
        inputs=circular_padded_inputs, training=False, mask=self._mask)
    self.assertAllClose(output_1, output_2)


if __name__ == '__main__':
  tf.test.main()
