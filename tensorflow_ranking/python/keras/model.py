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
"""Ranking Model utilities and classes in Keras."""

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import feature


def create_keras_model(network,
                       loss,
                       metrics,
                       optimizer,
                       size_feature_name,
                       list_size=None):
  """Creates a Functional Keras ranking model.

  A mask is inferred from size_feature_name and passed to the network, along
  with feature dictionary as inputs.

  Args:
    network: (`tfr.keras.network.RankingNetwork`) A ranking network which
      generates a list of scores.
    loss: (`tfr.keras.losses._RankingLoss`) A ranking loss.
    metrics: (list) List of ranking metrics, `tfr.keras.metrics._RankingMetric`
      instances.
    optimizer: (`tf.keras.optimizer.Optimizer`) Optimizer to minimize ranking
      loss.
    size_feature_name: (str) Name of feature for example list sizes. If not
      None, this feature name corresponds to a `tf.int32` Tensor of size
      [batch_size] corresponding to sizes of example lists. If `None`, all
      examples are treated as valid.
    list_size: (int) The list size for example features. If None, use dynamic
      list size. A fixed list size is required for TPU training.

  Returns:
    A compiled ranking Keras model, a `tf.keras.Model` instance.
  """
  # TODO: Support compatibility with TPUs.
  keras_inputs = feature.create_keras_inputs(
      context_feature_columns=network.context_feature_columns,
      example_feature_columns=network.example_feature_columns,
      size_feature_name=size_feature_name,
      list_size=list_size)

  # Create mask from sizes and list_size.
  mask = None
  if size_feature_name is not None:
    mask = feature.GenerateMask(network.example_feature_columns,
                                size_feature_name)(
                                    keras_inputs)
  logits = network(inputs=keras_inputs, mask=mask)

  ranker = tf.keras.Model(inputs=keras_inputs, outputs=logits)
  ranker.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return ranker
