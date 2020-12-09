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

  def __init__(self, name: Optional[str] = None, **kwargs: Dict[Any, Any]):
    """Initializes the FlattenList layer."""
    super().__init__(name=name, **kwargs)

  def call(
      self, context_features: Dict[str, tf.Tensor],
      example_features: Dict[str, tf.Tensor],
      mask: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Call FlattenList layer to flatten context_features and example_features.

    Args:
      context_features: A map of context features to 2D tensors of shape
        [batch_size, feature_dim].
      example_features: A map of example features to 3D tensors of shape
        [batch_size, list_size, feature_dim].
      mask: A Tensor of shape [batch_size, list_size] to mask out the invalid
        examples.

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
    batch_size = tf.shape(mask)[0]
    list_size = tf.shape(mask)[1]
    nd_indices, _ = utils.padded_nd_indices(is_valid=mask)

    # Expand context features to be of [batch_size, list_size, ...].
    flattened_context_features = {}
    for name, tensor in context_features.items():
      expanded_tensor = tf.repeat(
          tf.expand_dims(tensor, axis=1), repeats=[list_size], axis=1)
      flattened_context_features[name] = utils.reshape_first_ndims(
          expanded_tensor, 2, [batch_size * list_size])

    flattened_example_features = {}
    for name, tensor in example_features.items():
      # Replace invalid example features with valid ones.
      padded_tensor = tf.gather_nd(tensor, nd_indices)
      flattened_example_features[name] = utils.reshape_first_ndims(
          padded_tensor, 2, [batch_size * list_size])

    return flattened_context_features, flattened_example_features


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

  def call(self, flattened_logits: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Restores listwise shape of flattened_logits.

    Args:
      flattened_logits: A `Tensor` of predicted logits for each pair of query
        and documents, 1D tensor of shape [batch_size * list_size] or 2D tensor
        of shape [batch_size * list_size, 1].
      mask: A boolean `Tensor` of shape [batch_size, list_size] to mask out the
        invalid examples.

    Returns:
      A `Tensor` of shape [batch_size, list_size].

    Raises:
      ValueError: An error if the shape of `flattened_logits` is neither 1D nor
        2D with shape [batch_size * list_size, 1].
    """
    try:
      logits = tf.reshape(flattened_logits, shape=tf.shape(mask))
    except:
      raise ValueError('`flattened_logits` needs to be either 1D of batch_size '
                       '* list_size or 2D of [batch_size * list_size, 1].')
    if self._by_scatter:
      nd_indices, _ = utils.padded_nd_indices(is_valid=mask)
      counts = tf.scatter_nd(nd_indices, tf.ones_like(logits), tf.shape(mask))
      logits = tf.scatter_nd(nd_indices, logits, tf.shape(mask))
      return tf.where(
          tf.math.greater(counts, 0.), logits / counts, tf.math.log(_EPSILON))
    else:
      return tf.where(mask, logits, tf.math.log(_EPSILON))

    return logits

  def get_config(self):
    config = super().get_config()
    config.update({
        'by_scatter': self._by_scatter,
    })
    return config
