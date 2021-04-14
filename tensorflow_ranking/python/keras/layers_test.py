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
import tensorflow as tf

from tensorflow_ranking.python.keras import layers

_EPSILON = 1e-10


class TowerTest(tf.test.TestCase):

  def test_create_tower(self):
    inputs = tf.constant([[[1], [0], [-1]], [[0], [1], [0]]], dtype=tf.float32)
    tower = layers.create_tower([3, 2, 1], 1)
    outputs = tower(inputs)
    self.assertAllEqual([2, 3, 1], outputs.get_shape().as_list())


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
         inputs=(context_features, example_features, mask))
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
         inputs=(context_features, example_features, mask))
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
      layers.FlattenList()(inputs=(context_features, example_features, mask))

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
    logits = layers.RestoreList()(inputs=(self.flattened_logits, self.mask))
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllEqual([[1, 0.5, tf.math.log(_EPSILON)],
                         [0, tf.math.log(_EPSILON),
                          tf.math.log(_EPSILON)]], logits.numpy())
    logits = layers.RestoreList()(inputs=(self.flattened_logits_2d, self.mask))
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllEqual([[1, 0.5, tf.math.log(_EPSILON)],
                         [0, tf.math.log(_EPSILON),
                          tf.math.log(_EPSILON)]], logits.numpy())

  def test_call_by_scatter(self):
    logits = layers.RestoreList(by_scatter=True)(
        inputs=(self.flattened_logits, self.mask))
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllClose(
        [[1.5, 0.5, tf.math.log(_EPSILON)],
         [-1. / 3., tf.math.log(_EPSILON),
          tf.math.log(_EPSILON)]], logits.numpy())
    logits = layers.RestoreList(by_scatter=True)(
        inputs=(self.flattened_logits_2d, self.mask))
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

    logits = loaded(inputs=(self.flattened_logits, self.mask))
    self.assertAllEqual([2, 3], logits.get_shape().as_list())
    self.assertAllClose(
        [[1.5, 0.5, tf.math.log(_EPSILON)],
         [-1. / 3., tf.math.log(_EPSILON),
          tf.math.log(_EPSILON)]], logits.numpy())
    logits = loaded(inputs=(self.flattened_logits_2d, self.mask))
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
      layers.RestoreList()(inputs=(flattened_logits, mask))
    with self.assertRaises(ValueError):
      layers.RestoreList()(inputs=(flattened_logits_2d, mask))


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
        (context_features, example_features, mask))
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
        (context_features, example_features, mask))
    self.assertAllClose(expected_concat_tensor, concat_tensor)

  def test_serialization(self):
    layer = layers.ConcatFeatures()
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())


class DocumentInteractionAttentionLayerTest(tf.test.TestCase):

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
    self._dropout = 0.5

  def _get_din_layer(self):
    return layers.DocumentInteractionAttention(
        num_heads=self._num_heads,
        head_size=self._head_size,
        num_layers=self._num_layers,
        dropout=self._dropout)

  def test_serialization(self):
    # Check save and restore config.
    layer = self._get_din_layer()
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())

  def test_deterministic_inference_behavior(self):
    din_layer = self._get_din_layer()
    output_1 = din_layer(inputs=(self._inputs, self._mask), training=False)
    output_2 = din_layer(inputs=(self._inputs, self._mask), training=False)
    self.assertAllClose(output_1, output_2)

  def test_call(self):
    tf.random.set_seed(1)
    din_layer = self._get_din_layer()
    output = din_layer(inputs=(self._inputs, self._mask), training=False)
    self.assertEqual(output.shape.as_list(), [2, 3, self._head_size])

    expected_output = tf.convert_to_tensor([[[-1., 1.0000001], [-1., 1.0000001],
                                             [-1., 1.0000001]],
                                            [[-1., 0.99999994], [-1., 1.],
                                             [-1., 1.]]])
    self.assertAllClose(expected_output, output)

  def test_no_effect_circular_padding(self):
    din_layer = self._get_din_layer()
    output_1 = din_layer(inputs=(self._inputs, self._mask), training=False)
    circular_padded_inputs = tf.constant(
        [[[2., 1.], [2., 0.], [2., 1.]], [[1., 0.], [1., 0.], [1., 0.]]],
        dtype=tf.float32)
    output_2 = din_layer(
        inputs=(circular_padded_inputs, self._mask), training=False)
    self.assertAllClose(output_1, output_2)


class GAMLayerTest(tf.test.TestCase):

  def test_serialization(self):
    tf.random.set_seed(1)
    example_inputs = tf.constant([[1], [0], [-1]], dtype=tf.float32)
    context_inputs = tf.constant([[1, 2], [0, 1], [-1, 1]], dtype=tf.float32)
    layer = layers.GAMLayer(3, [3, 2, 1], 2, [3, 2, 1], activation=tf.nn.relu)
    serialized = tf.keras.layers.serialize(layer)
    loaded = tf.keras.layers.deserialize(serialized)
    self.assertAllEqual(loaded.get_config(), layer.get_config())

    outputs, _, _ = layer(([example_inputs, example_inputs, example_inputs
                           ], [context_inputs, context_inputs]),
                          training=False)
    self.assertAllClose([[-0.338468], [0.], [-0.340799]], outputs.numpy())

    outputs, _, _ = loaded(([example_inputs, example_inputs, example_inputs
                            ], [context_inputs, context_inputs]),
                           training=False)
    self.assertAllClose([[-0.016473], [0.], [-0.002832]], outputs.numpy())

  def test_gam_layer_call(self):
    example_inputs = tf.constant([[1], [0], [-1]], dtype=tf.float32)
    context_inputs = tf.constant([[1, 2], [0, 1], [-1, 1]], dtype=tf.float32)
    gam = layers.GAMLayer(2, [3, 2, 1], 2, [3, 2, 1])
    outputs, sublogits_list, subweights_list = gam(
        ([example_inputs, example_inputs], [context_inputs, context_inputs]))
    self.assertAllEqual([3, 1], outputs.get_shape().as_list())
    self.assertAllEqual(2, len(sublogits_list))
    for sublogits in sublogits_list:
      self.assertAllEqual([3, 1], sublogits.get_shape().as_list())
    self.assertAllEqual(2, len(subweights_list))
    for subweights in subweights_list:
      self.assertAllEqual([3, 2], subweights.get_shape().as_list())

  def test_gam_layer_call_without_context(self):
    example_inputs = tf.constant([[1], [0], [-1]], dtype=tf.float32)

    gam = layers.GAMLayer(2, [3, 2, 1], 2, [3, 2, 1])
    outputs, sublogits_list, subweights_list = gam(
        ([example_inputs, example_inputs], None))
    self.assertAllEqual([3, 1], outputs.get_shape().as_list())
    self.assertAllEqual(2, len(sublogits_list))
    for sublogits in sublogits_list:
      self.assertAllEqual([3, 1], sublogits.get_shape().as_list())
    self.assertAllEqual(0, len(subweights_list))

    gam_without_context = layers.GAMLayer(2, [3, 2, 1])
    outputs, sublogits_list, subweights_list = gam_without_context(
        ([example_inputs, example_inputs], None))
    self.assertAllEqual([3, 1], outputs.get_shape().as_list())
    self.assertAllEqual(2, len(sublogits_list))
    for sublogits in sublogits_list:
      self.assertAllEqual([3, 1], sublogits.get_shape().as_list())
    self.assertAllEqual(0, len(subweights_list))

  def test_gam_layer_call_raise_example_feature_num_error(self):
    example_inputs = tf.constant([[1], [0], [-1]], dtype=tf.float32)
    context_inputs = tf.constant([[1, 2], [0, 1], [-1, 1]], dtype=tf.float32)
    gam = layers.GAMLayer(3, [3, 2, 1], 2, [3, 2, 1])
    with self.assertRaises(ValueError):
      _ = gam(([example_inputs], [context_inputs, context_inputs]))

  def test_gam_layer_call_raise_context_feature_num_error(self):
    example_inputs = tf.constant([[1], [0], [-1]], dtype=tf.float32)
    context_inputs = tf.constant([[1, 2], [0, 1], [-1, 1]], dtype=tf.float32)
    gam = layers.GAMLayer(1, [3, 2, 1], 2, [3, 2, 1])
    with self.assertRaises(ValueError):
      _ = gam(([example_inputs], [context_inputs]))


if __name__ == '__main__':
  tf.test.main()
