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
"""Ranking Networks in Keras."""

import abc
import six
import tensorflow.compat.v2 as tf

from tensorflow_ranking.python import utils
from tensorflow_ranking.python.keras import feature


class RankingNetwork(tf.keras.layers.Layer):
  """Base class for ranking networks in Keras."""

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               name='ranking_network',
               **kwargs):
    """Initializes the base ranking network with feature columns.

    Args:
      context_feature_columns: (dict) context feature names to columns.
      example_feature_columns: (dict) example feature names to columns.
      name: (string) name of the model.
      **kwargs: keyword arguments.
    """
    super(RankingNetwork, self).__init__(name=name, **kwargs)
    self._context_feature_columns = context_feature_columns or {}
    self._example_feature_columns = example_feature_columns or {}
    self._listwise_dense_layer = feature.EncodeListwiseFeatures(
        context_feature_columns=self._context_feature_columns,
        example_feature_columns=self._example_feature_columns)

  @property
  def context_feature_columns(self):
    return self._context_feature_columns

  @property
  def example_feature_columns(self):
    return self._example_feature_columns

  def transform(self, features=None, training=None, mask=None):
    """Transforms the features into dense context features and example features.

    The user can overwrite this function for custom transformations.
    Mask is provided as an argument so that inherited models can have access
    to it for custom feature transformations, without modifying
    `call` explicitly.

    Args:
      features: (dict) with a mix of context (2D) and example features (3D).
      training: (bool) whether in train or inference mode.
      mask: (tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
        is True for a valid example and False for invalid one.

    Returns:
      context_features: (dict) context feature names to dense 2D tensors of
        shape [batch_size, feature_dims].
      example_features: (dict) example feature names to dense 3D tensors of
        shape [batch_size, list_size, feature_dims].
    """
    del mask
    context_features, example_features = self._listwise_dense_layer(
        inputs=features, training=training)
    return context_features, example_features

  @abc.abstractmethod
  def compute_logits(self,
                     context_features=None,
                     example_features=None,
                     training=None,
                     mask=None):
    """Scores context and examples to return a score per document.

    Args:
      context_features: (dict) context feature names to 2D tensors of shape
        [batch_size, feature_dims].
      example_features: (dict) example feature names to 3D tensors of shape
        [batch_size, list_size, feature_dims].
      training: (bool) whether in train or inference mode.
      mask: (tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
        is True for a valid example and False for invalid one. If mask is None,
        all entries are valid.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, list_size].
    """
    raise NotImplementedError('Calling an abstract method, '
                              'tfr.keras.RankingModel.compute_logits().')

  def call(self, inputs=None, training=None, mask=None):
    """Defines the forward pass for ranking model.

    Args:
      inputs: (dict) with a mix of context (2D) and example features (3D).
      training: (bool) whether in train or inference mode.
      mask: (tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
        is True for a valid example and False for invalid one.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, list_size].
    """
    context_features, example_features = self.transform(
        features=inputs, training=training, mask=mask)
    logits = self.compute_logits(
        context_features=context_features,
        example_features=example_features,
        training=training,
        mask=mask)
    return logits

  def get_config(self):
    config = super(RankingNetwork, self).get_config()
    config.update({
        'context_feature_columns':
            feature.serialize_feature_columns(self._context_feature_columns),
        'example_feature_columns':
            feature.serialize_feature_columns(self._example_feature_columns),
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates a RankingNetwork layer from its config.

    Args:
      config: (dict) Layer configuration, typically the output of `get_config`.
      custom_objects: (dict) Optional dictionary mapping names to custom classes
        or functions to be considered during deserialization.

    Returns:
      A RankingNetwork layer.
    """
    config_cp = config.copy()
    config_cp['context_feature_columns'] = feature.deserialize_feature_columns(
        config_cp['context_feature_columns'], custom_objects=custom_objects)
    config_cp['example_feature_columns'] = feature.deserialize_feature_columns(
        config_cp['example_feature_columns'], custom_objects=custom_objects)

    return cls(**config_cp)


class UnivariateRankingNetwork(RankingNetwork):
  """Base class for univariate ranking network."""

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               name='univariate_ranking_network',
               **kwargs):
    super(UnivariateRankingNetwork, self).__init__(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        name=name,
        **kwargs)

  @abc.abstractmethod
  def score(self, context_features=None, example_features=None, training=None):
    """Univariate scoring of context and one example to generate a score.

    Args:
      context_features: (dict) context feature names to 2D tensors of shape
        [batch_size, ...].
      example_features: (dict) example feature names to 2D tensors of shape
        [batch_size, ...].
      training: (bool) whether in training or inference mode.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, 1].
    """
    raise NotImplementedError('Calling an abstract method, '
                              'tfr.keras.UnivariateRankingModel.score().')

  def compute_logits(self,
                     context_features=None,
                     example_features=None,
                     training=None,
                     mask=None):
    """Scores context and examples to return a score per document.

    Args:
      context_features: (dict) context feature names to 2D tensors of shape
        [batch_size, feature_dims].
      example_features: (dict) example feature names to 3D tensors of shape
        [batch_size, list_size, feature_dims].
      training: (bool) whether in train or inference mode.
      mask: (tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
        is True for a valid example and False for invalid one. If mask is None,
        all entries are valid.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, list_size].
    """
    tensor = next(six.itervalues(example_features))
    batch_size = tf.shape(tensor)[0]
    list_size = tf.shape(tensor)[1]
    if mask is None:
      mask = tf.ones(shape=[batch_size, list_size], dtype=tf.bool)
    nd_indices, nd_mask = utils.padded_nd_indices(is_valid=mask)

    # Expand query features to be of [batch_size, list_size, ...].
    large_batch_context_features = {}
    for name, tensor in six.iteritems(context_features):
      x = tf.expand_dims(input=tensor, axis=1)
      x = tf.gather(x, tf.zeros([list_size], tf.int32), axis=1)
      large_batch_context_features[name] = utils.reshape_first_ndims(
          x, 2, [batch_size * list_size])

    large_batch_example_features = {}
    for name, tensor in six.iteritems(example_features):
      # Replace invalid example features with valid ones.
      padded_tensor = tf.gather_nd(tensor, nd_indices)
      large_batch_example_features[name] = utils.reshape_first_ndims(
          padded_tensor, 2, [batch_size * list_size])

    # Get scores for large batch.
    scores = self.score(
        context_features=large_batch_context_features,
        example_features=large_batch_example_features,
        training=training)
    logits = tf.reshape(
        scores, shape=[batch_size, list_size])

    # Apply nd_mask to zero out invalid entries.
    logits = tf.where(nd_mask, logits, tf.zeros_like(logits))
    return logits
