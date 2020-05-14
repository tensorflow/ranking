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
"""Generalized Additive Model (GAM) Keras Network."""

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import network as network_lib

# Postfix for names of subscore tensors.
_SUBSCORE_POSTFIX = 'subscore'

# Postfix for names of subweight tensors.
_SUBWEIGHT_POSTFIX = 'subweight'


def _make_tower_layers(hidden_layer_dims,
                       output_units,
                       activation=None,
                       use_batch_norm=True,
                       batch_norm_moment=0.999,
                       dropout=0.5):
  """Defines tower using keras layers.

  Args:
   hidden_layer_dims: Iterable of number hidden units per layer.
     All layers are fully connected. Ex. `[64, 32]` means first layer has 64
     nodes and second one has 32.
   output_units: (int) Size of output logits from this tower.
   activation: Activation function applied to each layer. If `None`, will use
        an identity activation, which is default behavior in Keras activations.
   use_batch_norm: Whether to use batch normalization after each hidden layer.
   batch_norm_moment: Momentum for the moving average in batch normalization.
   dropout: When not `None`, the probability we will drop out a given
      coordinate.

  Returns:
    A list of Keras layers for this tower.
  """
  layers = []
  if not hidden_layer_dims:
    return layers
  if use_batch_norm:
    layers.append(
        tf.keras.layers.BatchNormalization(momentum=batch_norm_moment))
  for layer_width in hidden_layer_dims:
    layers.append(tf.keras.layers.Dense(units=layer_width))
    if use_batch_norm:
      layers.append(
          tf.keras.layers.BatchNormalization(momentum=batch_norm_moment))
    layers.append(tf.keras.layers.Activation(activation=activation))
    if dropout:
      layers.append(tf.keras.layers.Dropout(rate=dropout))
  layers.append(tf.keras.layers.Dense(units=output_units))
  return layers


class GAMRankingNetwork(network_lib.UnivariateRankingNetwork):
  """Generalized Additive Model (GAM) based univariate ranking network."""

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               context_hidden_layer_dims=None,
               example_hidden_layer_dims=None,
               activation=None,
               use_batch_norm=True,
               batch_norm_moment=0.999,
               dropout=0.5,
               name='gam_ranking_model',
               **kwargs):
    """Initializes an instance of `GAMRankingNetwork`.

    Args:
      context_feature_columns: A dict containing all the context feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      example_feature_columns: A dict containing all the example feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      context_hidden_layer_dims: Iterable of number hidden units per layer for
        context features. See `example_hidden_units`.
      example_hidden_layer_dims: Iterable of number hidden units per layer for
        example features. All layers are fully connected. Ex. `[64, 32]` means
        first layer has 64 nodes and second one has 32.
      activation: Activation function applied to each layer. If `None`, will use
        an identity activation, which is default behavior in Keras activations.
      use_batch_norm: Whether to use batch normalization after each hidden
        layer.
      batch_norm_moment: Momentum for the moving average in batch normalization.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      name: name of the keras network.
      **kwargs: Keyword arguments.

    Raises:
       `ValueError` if `example_feature_columns` is empty or if
       `example_hidden_lyaer_dims` is empty.
    """
    if not example_feature_columns or not example_hidden_layer_dims:
      raise ValueError('example_feature_columns or example_hidden_layer_dims '
                       'must not be empty.')
    super(GAMRankingNetwork, self).__init__(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        name=name,
        **kwargs)
    context_hidden_layer_dims = context_hidden_layer_dims or []
    self._context_hidden_layer_dims = [
        int(d) for d in context_hidden_layer_dims
    ]
    self._example_hidden_layer_dims = [
        int(d) for d in example_hidden_layer_dims
    ]
    self._num_features = len(self.example_feature_columns)

    self._activation = activation
    self._use_batch_norm = use_batch_norm
    self._batch_norm_moment = batch_norm_moment
    self._dropout = dropout

    self._per_context_feature_layers = {}
    for name in self._context_feature_columns:
      self._per_context_feature_layers[name] = _make_tower_layers(
          hidden_layer_dims=self._context_hidden_layer_dims,
          output_units=self._num_features,
          activation=self._activation,
          use_batch_norm=self._use_batch_norm,
          batch_norm_moment=self._batch_norm_moment,
          dropout=self._dropout)

    self._per_example_feature_layers = {}
    for name in self._example_feature_columns:
      self._per_example_feature_layers[name] = _make_tower_layers(
          hidden_layer_dims=self._example_hidden_layer_dims,
          output_units=1,
          activation=self._activation,
          use_batch_norm=self._use_batch_norm,
          batch_norm_moment=self._batch_norm_moment,
          dropout=self._dropout)

  def score(self, context_features=None, example_features=None, training=True):
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
    example_feature_names = sorted(list(self.example_feature_columns.keys()))
    context_feature_names = sorted(list(self.context_feature_columns.keys()))
    context_input = [
        tf.keras.layers.Flatten()(context_features[name])
        for name in context_feature_names
    ]
    example_input = [
        tf.keras.layers.Flatten()(example_features[name])
        for name in example_feature_names
    ]

    # Construct a tower for each example feature.
    sub_logits_list = []
    with tf.name_scope('example_feature_towers'):
      for name, input_tensor in zip(example_feature_names, example_input):
        with tf.name_scope('{}_tower'.format(name)):
          cur = input_tensor
          layers = self._per_example_feature_layers[name]
          for layer in layers:
            cur = layer(cur, training=training)
          sub_logits = tf.identity(
              cur, name='{}_{}'.format(name, _SUBSCORE_POSTFIX))
          sub_logits_list.append(sub_logits)

    sub_weights_list = []
    if context_input:
      # Construct a tower for each context feature.
      with tf.name_scope('context_feature_towers'):
        for name, input_tensor in zip(context_feature_names, context_input):
          with tf.name_scope('{}_tower'.format(name)):
            cur = input_tensor
            layers = self._per_context_feature_layers[name]
            for layer in layers:
              cur = layer(cur, training=training)
            cur = tf.keras.layers.Softmax()(cur)
            sub_weights = tf.identity(
                cur, name='{}_{}'.format(name, _SUBWEIGHT_POSTFIX))
            sub_weights_list.append(sub_weights)

    # Construct an additive model from the outputs of all example feature towers
    # weighted by outputs of all context feature towers.
    # Note that these layers do not have any trainable variables, hence we
    # are not defining them in init but defining them here, similar to Flatten.
    if sub_weights_list:
      sub_logits = tf.keras.layers.Concatenate(axis=-1)(sub_logits_list)
      sub_weights = (
          tf.keras.layers.Add()(sub_weights_list)
          if len(sub_weights_list) > 1 else sub_weights_list[0])
      logits = tf.keras.backend.sum(sub_logits * sub_weights, axis=-1)
    else:
      logits = tf.keras.layers.Add()(
          sub_logits_list) if len(sub_logits_list) > 1 else sub_logits_list[0]
    return logits

  def get_config(self):
    config = super(GAMRankingNetwork, self).get_config()
    config.update({
        'context_hidden_layer_dims': self._context_hidden_layer_dims,
        'example_hidden_layer_dims': self._example_hidden_layer_dims,
        'activation': self._activation,
        'use_batch_norm': self._use_batch_norm,
        'batch_norm_moment': self._batch_norm_moment,
        'dropout': self._dropout,
    })
    return config
