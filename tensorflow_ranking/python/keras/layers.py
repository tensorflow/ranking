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
"""Defines Keras Layers for TF-Ranking."""

from typing import Any, Dict, Optional, Tuple
import tensorflow as tf

from official.nlp.modeling import layers as nlp_modeling_layers
from tensorflow_ranking.python import utils

_EPSILON = 1e-10


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class FlattenList(tf.keras.layers.Layer):
  """Layer to flatten the example list.

  This layer flattens the batch_size dimension and the list_size dimension for
  the `example_features` and expands list_size times for the `context_features`.

  Example use,
  ```python
    context_features = {'context_feature_1': [[1], [0]]}
    example_features = {'example_feature_1':
                        [[[1], [0], [-1]], [[0], [1], [0]]]}
    mask = [[True, True, False], [True, False, False]]
    flattened_context_features, flattened_example_features = FlattenList()(
        context_features, example_features, mask)
  ```
  That is, there are two valid examples in the first query and one
  valid example in the second query. Then
  ```python
    flattened_context_features = {'context_feature_1':
                                  [[1], [1], [1], [0], [0], [0]]}
    flattened_example_features = {'example_feature_1':
                                  [[1], [0], [1], [0], [0], [0]]}
  ```
  `context_feature_1` is repeated by list_size=3 times. `example_feature_1` is
  flattened and padded with the invalid terms replaced by valid terms in each
  query in a circular way.
  """

  def __init__(self,
               circular_padding: bool = True,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the FlattenList layer.

    Args:
      circular_padding: Whether to apply circular padding to replace invalid
        features with valid ones.
      name: Name of the layer.
      **kwargs: keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._circular_padding = circular_padding

  def call(
      self, context_features: Dict[str, tf.Tensor],
      example_features: Dict[str, tf.Tensor], list_mask: tf.Tensor
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Call FlattenList layer to flatten context_features and example_features.

    Args:
      context_features: A map of context features to 2D tensors of shape
        [batch_size, feature_dim].
      example_features: A map of example features to 3D tensors of shape
        [batch_size, list_size, feature_dim].
      list_mask: A Tensor of shape [batch_size, list_size] to mask out the
        invalid examples.

    Returns:
      A tuple of (flattened_context_features, flattened_example_fatures) where
      the former is a dict of context features to 2D tensors of shape
      [batch_size * list_size, feature_dim] and the latter is a dict of example
      features to 2D tensors of shape [batch_size * list_size, feature_dim].

    Raises:
      ValueError: An error if example_features is None or empty.
    """
    if not example_features:
      raise ValueError('Need a valid example feature.')
    batch_size = tf.shape(list_mask)[0]
    list_size = tf.shape(list_mask)[1]
    # Expand context features to be of [batch_size, list_size, ...].
    flattened_context_features = {}
    for name, tensor in context_features.items():
      expanded_tensor = tf.repeat(
          tf.expand_dims(tensor, axis=1), repeats=[list_size], axis=1)
      flattened_context_features[name] = utils.reshape_first_ndims(
          expanded_tensor, 2, [batch_size * list_size])

    nd_indices = None
    if self._circular_padding:
      nd_indices, _ = utils.padded_nd_indices(is_valid=list_mask)

    flattened_example_features = {}
    for name, tensor in example_features.items():
      if nd_indices is not None:
        # Replace invalid example features with valid ones.
        tensor = tf.gather_nd(tensor, nd_indices)
      flattened_example_features[name] = utils.reshape_first_ndims(
          tensor, 2, [batch_size * list_size])

    return flattened_context_features, flattened_example_features

  def get_config(self):
    config = super().get_config()
    config.update({
        'circular_padding': self._circular_padding,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class RestoreList(tf.keras.layers.Layer):
  """Output layer to restore listwise output shape.

  This layer maps the flattened 1D logits or 2D tensor of shape
  [batch_size * list_size, 1] back to 2D of shape [batch_size, list_size] and
  mask the invalid terms to be a defined large negative value.

  Example use,
  ```python
    flattened_logits = [1, 0.5, 2, 0, -1, 0]
    mask = [[True, True, False], [True, False, False]]
    logits = RestoreList()(flattened_logits, mask)
  ```
  Then
  ```python
    logits = [[1, 0.5, log(_EPSILON)], [0, log(_EPSILON), log(_EPSILON)]]
  ```
  where _EPSILON=1e-10. This layer works also for 2D `flattened_logits` like
  [[1], [0.5], [2], [0], [-1], [0]].

  When `by_scatter=True`, an nd_indices will be generated using `mask` in the
  same way as `FlattenList`. All values in the `flattened_logits` will be used
  and repeated entries will be averaged.
  ```python
    flattened_logits = [1, 0.5, 2, 0, -1, 0]
    mask = [[True, True, False], [True, False, False]]
    logits = RestoreList(by_scatter=True)(flattened_logits, mask)
  ```
  Then
  ```python
    logits = [[1.5, 0.5, log(_EPSILON)], [-1/3, log(_EPSILON), log(_EPSILON)]]
  ```
  This is because the flattened_logits are treated as circularly padded entries.
  The [1st, 3rd] values [1, 2] are counted to logits[0, 0]. The [4th, 5th, 6th]
  values [0, -1, 0] are counted to logits[1, 0]. Note that We use different
  values for those repeated entries, while they are likely the same in practice.
  """

  def __init__(self,
               name: Optional[str] = None,
               by_scatter: bool = False,
               **kwargs: Dict[Any, Any]):
    super().__init__(name=name, **kwargs)
    self._by_scatter = by_scatter

  def call(self, flattened_logits: tf.Tensor,
           list_mask: tf.Tensor) -> tf.Tensor:
    """Restores listwise shape of flattened_logits.

    Args:
      flattened_logits: A `Tensor` of predicted logits for each pair of query
        and documents, 1D tensor of shape [batch_size * list_size] or 2D tensor
        of shape [batch_size * list_size, 1].
      list_mask: A boolean `Tensor` of shape [batch_size, list_size] to mask out
        the invalid examples.

    Returns:
      A `Tensor` of shape [batch_size, list_size].

    Raises:
      ValueError: An error if the shape of `flattened_logits` is neither 1D nor
        2D with shape [batch_size * list_size, 1].
    """
    try:
      logits = tf.reshape(flattened_logits, shape=tf.shape(list_mask))
    except:
      raise ValueError('`flattened_logits` needs to be either '
                       '1D of [batch_size * list_size] or '
                       '2D of [batch_size * list_size, 1].')
    if self._by_scatter:
      nd_indices, _ = utils.padded_nd_indices(is_valid=list_mask)
      counts = tf.scatter_nd(nd_indices, tf.ones_like(logits),
                             tf.shape(list_mask))
      logits = tf.scatter_nd(nd_indices, logits, tf.shape(list_mask))
      return tf.where(
          tf.math.greater(counts, 0.), logits / counts, tf.math.log(_EPSILON))
    else:
      return tf.where(list_mask, logits, tf.math.log(_EPSILON))

  def get_config(self):
    config = super().get_config()
    config.update({
        'by_scatter': self._by_scatter,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ConcatFeatures(tf.keras.layers.Layer):
  """Concatenates context features and example features in a listwise manner.

  Given dicts of `context features`, `example features`, this layer expands
  list_size times for the `context_features` and concatenates them with
  example features along the `list_size` axis. The output is a 3-d tensor with
  shape [batch_size, list_size, sum(feature_dims)], where sum(feature_dims) is
  the sum of all example feature dimensions and the context feature dimension.

  Example:
  ```python
    # Batch size = 2, list_size = 2.
    context_features = {
        'context_feature_1': [[1.], [2.]]
    }
    example_features = {
        'example_feature_1':
            [[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]
    }
    mask = [[True, False], [True, True]]
    ConcatFeatures()(context_features, example_features, mask)
    # Returns: [[[1., 1., 0.], [1., 1., 0.]], [[2., 0., 1.], [2., 1., 0.]]])

    ConcatFeatures(circular_padding=False)(
        context_features, example_features, mask)
    # Returns: [[[1., 1., 0.], [1., 0., 1.]], [[2., 0., 1.], [2., 1., 0.]]]
  ```
  """

  def __init__(self,
               circular_padding: bool = True,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the ConcatFeatures layer.

    Args:
      circular_padding: Whether to apply circular padding to replace invalid
        features with valid ones.
      name: Name of the layer.
      **kwargs: keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._circular_padding = circular_padding
    self._flatten_list = FlattenList(circular_padding=self._circular_padding)

  def call(
      self,
      context_features: Dict[str, tf.Tensor],
      example_features: Dict[str, tf.Tensor],
      list_mask: [tf.Tensor],
  ) -> tf.Tensor:
    """Call method for ConcatFeatures layer.

    Args:
      context_features: A dict of `Tensor`s with shape [batch_size, ...].
      example_features:  A dict of `Tensor`s with shape [batch_size, list_size,
        ...].
      list_mask: A boolean tensor of shape [batch_size, list_size], which is
        True for a valid example and False for invalid one.

    Returns:
      A `Tensor` of shape [batch_size, list_size, ...].
    """
    (flattened_context_features,
     flattened_example_features) = self._flatten_list(
         context_features=context_features,
         example_features=example_features,
         list_mask=list_mask)
    # Concatenate flattened context and example features along `list_size` dim.
    context_input = [
        tf.keras.layers.Flatten()(flattened_context_features[name])
        for name in sorted(flattened_context_features)
    ]
    example_input = [
        tf.keras.layers.Flatten()(flattened_example_features[name])
        for name in sorted(flattened_example_features)
    ]
    flattened_concat_features = tf.concat(context_input + example_input, 1)

    # Reshape to 3D.
    batch_size = tf.shape(list_mask)[0]
    list_size = tf.shape(list_mask)[1]
    return utils.reshape_first_ndims(flattened_concat_features, 1,
                                     [batch_size, list_size])

  def get_config(self):
    config = super().get_config()
    config.update({
        'circular_padding': self._circular_padding,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class DocumentInteractionAttention(tf.keras.layers.Layer):
  """Cross Document Interaction Attention layer.

  This layer implements the cross-document attention described in
  "Permutation Equivariant Document Interaction Network for Neural Learning to
  Rank". http://research.google/pubs/pub49364/

  This layer comprises of several layers of Multi-Headed Attention (MHA)
  applied over the list of documents to attend over itself, using a mask to
  specify valid documents. Optionally, the user can specify the `topk` documents
  as the listwise context that is used to compute the attention per document. If
  `topk` is None, all the documents are used as listwise context to compute the
  attention.

  MHA uses scaled dot product attention, with residual connection and layer
  normalization as follows. This transformation is applied for `num_layers`
  times:
  h_i := LayerNorm_i(h_{i-1} + MHA_i(h_{i-1}), TopK(h_{i-1}; k))

  Example:
  ```python
    # Batch size = 2, list_size = 3.
    inputs =  [[[1., 1.], [1., 0.], [1., 1.]], [[0., 0.], [0., 0.], [0., 0.]]]
    list_mask = [[True, True, False], [True, False, False]]
    dia_layer = DocumentInteractionAttention(
        num_heads=1, head_size=64, num_layers=1, topk=1)
    dia_output = dia_layer(
        inputs=inputs,
        training=False,
        list_mask=list_mask)
  ```
  """

  def __init__(self,
               num_heads: int,
               head_size: int,
               num_layers: int = 1,
               dropout_rate: float = 0.5,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the layer.

    Args:
      num_heads: Number of attention heads (see `MultiHeadAttention` for more
        details on this argument).
      head_size: Size of each attention head.
      num_layers: Number of cross-document attention layers.
      dropout_rate: Dropout probability.
      name: Name of the layer.
      **kwargs: keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._num_heads = num_heads
    self._head_size = head_size
    self._num_layers = num_layers
    self._dropout_rate = dropout_rate

    # This projects input to head_size, so that this layer can be applied
    # recursively for `num_layers` times.
    # Shape: [batch_size, list_size, feature_dims] ->
    # [batch_size, list_size, head_size].
    self._input_projection = tf.keras.layers.Dense(
        units=self._head_size, activation='relu')

    # Self-attention layers.
    self._attention_layers = []
    for _ in range(self._num_layers):
      # Shape: [batch_size, list_size, head_size] ->
      # [batch_size, list_size, head_size].
      attention_layer = tf.keras.layers.MultiHeadAttention(
          num_heads=self._num_heads,
          key_dim=self._head_size,
          dropout=self._dropout_rate,
          output_shape=self._head_size)

      # Dropout and layer normalization are applied element-wise, and do not
      # change the shape.
      dropout_layer = tf.keras.layers.Dropout(rate=self._dropout_rate)
      norm_layer = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, dtype=tf.float32)
      self._attention_layers.append(
          (attention_layer, dropout_layer, norm_layer))

  def call(self,
           inputs: tf.Tensor,
           training: bool = True,
           list_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Calls the document interaction layer to apply cross-document attention.

    Args:
      inputs: A tensor of shape [batch_size, list_size, feature_dims].
      training: Whether in training or inference mode.
      list_mask: A boolean tensor of shape [batch_size, list_size], which is
        True for a valid example and False for invalid one. If this is `None`,
        then all examples are treated as valid.

    Returns:
      A tensor of shape [batch_size, list_size, head_size].
    """
    batch_size = tf.shape(inputs)[0]
    list_size = tf.shape(inputs)[1]
    if list_mask is None:
      list_mask = tf.ones(shape=(batch_size, list_size), dtype=tf.bool)
    input_tensor = self._input_projection(inputs, training=training)

    list_mask = tf.cast(list_mask, dtype=tf.int32)
    attention_mask = nlp_modeling_layers.SelfAttentionMask()(
        [list_mask, list_mask])

    for attention_layer, dropout_layer, norm_layer in self._attention_layers:
      output = attention_layer(
          query=input_tensor,
          value=input_tensor,
          attention_mask=attention_mask,
          training=training)
      output = dropout_layer(output, training=training)
      # Applying residual network here, similar to logic in Transformer.
      input_tensor = norm_layer(output + input_tensor, training=training)

    return input_tensor

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_heads': self._num_heads,
        'head_size': self._head_size,
        'num_layers': self._num_layers,
        'dropout_rate': self._dropout_rate,
    })
    return config
