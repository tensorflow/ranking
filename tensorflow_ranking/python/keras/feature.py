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
"""Feature transformations for ranking in Keras."""

import six
import tensorflow.compat.v2 as tf

from tensorflow_ranking.python import utils


def create_keras_inputs(context_feature_columns,
                        example_feature_columns,
                        size_feature_name):
  """Create Keras inputs from context and example feature columns.

  Args:
    context_feature_columns: (dict) context feature names to columns.
    example_feature_columns: (dict) example feature names to columns.
    size_feature_name: (str) Name of feature for example list sizes. If not
      None, this feature name corresponds to a `tf.int32` Tensor of size
      [batch_size] corresponding to sizes of example lists.

  Returns:
    A dict mapping feature names to Keras Input tensors.
  """
  context_feature_spec = tf.feature_column.make_parse_example_spec(
      context_feature_columns.values())
  example_feature_spec = tf.feature_column.make_parse_example_spec(
      example_feature_columns.values())

  feature_spec = {}
  feature_spec.update(context_feature_spec)
  feature_spec.update(example_feature_spec)

  inputs_dict = {}
  # Create Keras inputs for context features.
  for name, spec in six.iteritems(context_feature_spec):
    is_sparse = isinstance(spec, tf.io.VarLenFeature)
    shape = spec.shape if not is_sparse else (1,)
    inputs_dict[name] = tf.keras.Input(
        name=name, shape=shape, dtype=spec.dtype, sparse=is_sparse)

  # Create Keras inputs for example features. We set list_size to None to allow
  # for dynamic list sizes during inference.
  list_size = None
  for name, spec in six.iteritems(example_feature_spec):
    is_sparse = isinstance(spec, tf.io.VarLenFeature)
    shape = [list_size] + list(spec.shape) if not is_sparse else (1,)
    inputs_dict[name] = tf.keras.Input(
        name=name, shape=shape, dtype=spec.dtype, sparse=is_sparse)

  # Add size feature information.
  if size_feature_name is not None:
    inputs_dict[size_feature_name] = tf.keras.Input(
        name=size_feature_name, shape=(), dtype=tf.int32)
  return inputs_dict


class GenerateMask(tf.keras.layers.Layer):
  """Layer to generate mask."""

  def __init__(self,
               example_feature_columns,
               size_feature_name,
               name='generate_mask_layer',
               **kwargs):
    """Constructs a mask generator layer.

    Args:
      example_feature_columns: (dict) example feature names to columns.
      size_feature_name: (str) Name of feature for example list sizes. If not
        None, this feature name corresponds to a `tf.int32` Tensor of size
        [batch_size] corresponding to sizes of example lists. If `None`, all
        examples are treated as valid.
      name: (str) name of the layer.
      **kwargs: keyword arguments.
    """
    super(GenerateMask, self).__init__(name=name, **kwargs)
    self._example_feature_columns = example_feature_columns
    self._size_feature_name = size_feature_name

  @property
  def example_feature_columns(self):
    return self._example_feature_columns

  @property
  def size_feature_name(self):
    return self._size_feature_name

  def call(self, inputs):
    """Generates mask (whether example is valid) from features.

    Args:
      inputs: (dict) Features with a mix of context (2D) and example features
        (3D).

    Returns:
      mask: (tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
        is True for a valid example and False for invalid one.
    """
    example_feature = inputs[next(six.iterkeys(self._example_feature_columns))]
    list_size = tf.shape(example_feature)[1]
    sizes = inputs[self._size_feature_name]
    mask = tf.sequence_mask(sizes, maxlen=list_size)
    return mask

  def get_config(self):
    config = super(GenerateMask, self).get_config()
    config.update({
        'example_feature_columns': self._example_feature_columns,
        'size_feature_name': self._size_feature_name
    })
    return config


class EncodeListwiseFeatures(tf.keras.layers.Layer):
  """A layer that produces dense `Tensors` from context and example features."""

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               name='listwise_dense_features',
               **kwargs):
    """Constructs a ListwiseDenseFeatures layer.

    Args:
     context_feature_columns: (dict) context feature names to columns.
     example_feature_columns: (dict) example feature names to columns.
     name: (str) name of the layer.
     **kwargs: keyword arguments.
    """
    super(EncodeListwiseFeatures, self).__init__(name=name, **kwargs)
    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    if self._context_feature_columns:
      self._context_dense_layer = tf.keras.layers.DenseFeatures(
          feature_columns=six.itervalues(self._context_feature_columns),
          name=name)
    self._example_dense_layer = tf.keras.layers.DenseFeatures(
        feature_columns=six.itervalues(self._example_feature_columns),
        name=name)

  @property
  def context_feature_columns(self):
    return self._context_feature_columns

  @property
  def example_feature_columns(self):
    return self._example_feature_columns

  def call(self, inputs, training=True):
    """Transforms the features into dense context features and example features.

    This is the Keras equivalent of `tfr.feature.encode_listwise_features`.

    Args:
      inputs: (dict) Features with a mix of context (2D) and example features
        (3D).
      training: (bool) whether in train or inference mode.

    Returns:
      context_features: (dict) context feature names to dense 2D tensors of
        shape [batch_size, feature_dims].
      example_features: (dict) example feature names to dense 3D tensors of
        shape [batch_size, list_size, feature_dims].
    """
    features = inputs
    context_features = {}
    if self._context_feature_columns:
      context_cols_to_tensors = {}
      self._context_dense_layer(
          features,
          training=training,
          cols_to_output_tensors=context_cols_to_tensors)
      context_features = {
          name: context_cols_to_tensors[col]
          for name, col in six.iteritems(self.context_feature_columns)
      }
    example_features = {}
    if self._example_feature_columns:
      # Compute example_features. Note that the key in `example_feature_columns`
      # dict can be different from the key in the `features` dict. We only need
      # to reshape the per-example tensors in `features`. To obtain the keys for
      # per-example features, we use the parsing feature specs.
      example_specs = tf.feature_column.make_parse_example_spec(
          list(six.itervalues(self._example_feature_columns)))
      example_name = next(six.iterkeys(example_specs))
      batch_size = tf.shape(input=features[example_name])[0]
      list_size = tf.shape(input=features[example_name])[1]
      reshaped_example_features = {}
      for name in example_specs:
        if name not in features:
          continue
        reshaped_example_features[name] = utils.reshape_first_ndims(
            features[name], 2, [batch_size * list_size])

      example_cols_to_tensors = {}
      self._example_dense_layer(
          reshaped_example_features,
          training=training,
          cols_to_output_tensors=example_cols_to_tensors)
      example_features = {
          name: utils.reshape_first_ndims(example_cols_to_tensors[col], 1,
                                          [batch_size, list_size])
          for name, col in six.iteritems(self._example_feature_columns)
      }
    return context_features, example_features

  def get_config(self):
    config = super(EncodeListwiseFeatures, self).get_config()

    config.update({
        'context_feature_columns': self._context_feature_columns,
        'example_feature_columns': self._example_feature_columns,
    })
    return config
