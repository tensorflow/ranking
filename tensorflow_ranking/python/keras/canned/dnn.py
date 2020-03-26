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
"""DNN Ranking network in Keras."""

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import network as network_lib


class DNNRankingNetwork(network_lib.UnivariateRankingNetwork):
  """Deep Neural Network (DNN) scoring based univariate ranking network."""

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               hidden_layer_dims=None,
               activation=None,
               use_batch_norm=True,
               batch_norm_moment=0.999,
               dropout=0.5,
               name='dnn_ranking_network',
               **kwargs):
    """Initializes an instance of DNN ranking network.

    This network consists of feedforward linear units passed through a
    non-linear
    activation. The hidden size of the linear units and the activation are
    specified by the user.

    Args:
      context_feature_columns: A dict containing all the context feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      example_feature_columns: A dict containing all the example feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      hidden_layer_dims: Iterable of number hidden units per layer. All layers
        are fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
        second one has 32.
      activation: Activation function applied to each layer. If `None`, will use
        an identity activation, which is default behavior in Keras activations.
      use_batch_norm: Whether to use batch normalization after each hidden
        layer.
      batch_norm_moment: Momentum for the moving average in batch normalization.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      name: name of Keras network.
      **kwargs: keyword arguments.

    Raises:
      `ValueError` if `example_feature_columns` or `hidden_layer_dims` is empty.

    """
    if not example_feature_columns or not hidden_layer_dims:
      raise ValueError('example_feature_columns or hidden_layer_dims must not '
                       'be empty.')
    super(DNNRankingNetwork, self).__init__(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        name=name,
        **kwargs)
    self._hidden_layer_dims = [int(d) for d in hidden_layer_dims]
    self._activation = activation
    self._use_batch_norm = use_batch_norm
    self._batch_norm_moment = batch_norm_moment
    self._dropout = dropout

    layers = []
    if self._use_batch_norm:
      layers.append(
          tf.keras.layers.BatchNormalization(momentum=self._batch_norm_moment))
    for _, layer_width in enumerate(self._hidden_layer_dims):
      layers.append(tf.keras.layers.Dense(units=layer_width))
      if self._use_batch_norm:
        layers.append(
            tf.keras.layers.BatchNormalization(
                momentum=self._batch_norm_moment))
      layers.append(tf.keras.layers.Activation(activation=self._activation))
      layers.append(tf.keras.layers.Dropout(rate=self._dropout))

    self._scoring_layers = layers

    self._output_score_layer = tf.keras.layers.Dense(units=1)

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
    context_input = [
        tf.keras.layers.Flatten()(context_features[name])
        for name in sorted(self.context_feature_columns)
    ]
    example_input = [
        tf.keras.layers.Flatten()(example_features[name])
        for name in sorted(self.example_feature_columns)
    ]

    inputs = tf.concat(context_input + example_input, 1)

    outputs = inputs
    for layer in self._scoring_layers:
      outputs = layer(outputs, training=training)

    return self._output_score_layer(outputs, training=training)

  def get_config(self):
    config = super(DNNRankingNetwork, self).get_config()
    config.update({
        'hidden_layer_dims': self._hidden_layer_dims,
        'activation': self._activation,
        'use_batch_norm': self._use_batch_norm,
        'batch_norm_moment': self._batch_norm_moment,
        'dropout': self._dropout,
    })
    return config
