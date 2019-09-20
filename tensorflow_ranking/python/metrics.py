# Copyright 2019 The TensorFlow Ranking Authors.
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

"""Defines ranking metrics as TF ops.

The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tensorflow_ranking.python import utils


class RankingMetricKey(object):
  """Ranking metric key strings."""
  # Mean Receiprocal Rank. For binary relevance.
  MRR = 'mrr'

  # Average Relvance Position.
  ARP = 'arp'

  # Normalized Discounted Culmulative Gain.
  NDCG = 'ndcg'

  # Discounted Culmulative Gain.
  DCG = 'dcg'

  # Precision. For binary relevance.
  PRECISION = 'precision'

  # Ordered Pair Accuracy.
  ORDERED_PAIR_ACCURACY = 'ordered_pair_accuracy'


def make_ranking_metric_fn(metric_key,
                           weights_feature_name=None,
                           topn=None,
                           name=None):
  """Factory method to create a ranking metric function.

  Args:
    metric_key: A key in `RankingMetricKey`.
    weights_feature_name: A `string` specifying the name of the weights feature
      in `features` dict.
    topn: An `integer` specifying the cutoff of how many items are considered in
      the metric.
    name: A `string` used as the name for this metric.

  Returns:
    A metric fn with the following Args:
    * `labels`: A `Tensor` of the same shape as `predictions` representing
    graded relevance.
    * `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
    is the ranking score of the corresponding example.
    * `features`: A dict of `Tensor`s that contains all features.
  """

  def _get_weights(features):
    """Get weights tensor from features and reshape it to 2-D if necessary."""
    weights = None
    if weights_feature_name:
      weights = tf.convert_to_tensor(value=features[weights_feature_name])
      # Convert weights to a 2-D Tensor.
      weights = utils.reshape_to_2d(weights)
    return weights

  def _average_relevance_position_fn(labels, predictions, features):
    """Returns average relevance position as the metric."""
    return average_relevance_position(
        labels, predictions, weights=_get_weights(features), name=name)

  def _mean_reciprocal_rank_fn(labels, predictions, features):
    """Returns mean reciprocal rank as the metric."""
    return mean_reciprocal_rank(
        labels, predictions, weights=_get_weights(features), name=name)

  def _normalized_discounted_cumulative_gain_fn(labels, predictions, features):
    """Returns normalized discounted cumulative gain as the metric."""
    return normalized_discounted_cumulative_gain(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _discounted_cumulative_gain_fn(labels, predictions, features):
    """Returns discounted cumulative gain as the metric."""
    return discounted_cumulative_gain(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _precision_fn(labels, predictions, features):
    """Returns precision as the metric."""
    return precision(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _ordered_pair_accuracy_fn(labels, predictions, features):
    """Returns ordered pair accuracy as the metric."""
    return ordered_pair_accuracy(
        labels, predictions, weights=_get_weights(features), name=name)

  metric_fn_dict = {
      RankingMetricKey.ARP: _average_relevance_position_fn,
      RankingMetricKey.MRR: _mean_reciprocal_rank_fn,
      RankingMetricKey.NDCG: _normalized_discounted_cumulative_gain_fn,
      RankingMetricKey.DCG: _discounted_cumulative_gain_fn,
      RankingMetricKey.PRECISION: _precision_fn,
      RankingMetricKey.ORDERED_PAIR_ACCURACY: _ordered_pair_accuracy_fn,
  }
  assert metric_key in metric_fn_dict, ('metric_key %s not supported.' %
                                        metric_key)
  return metric_fn_dict[metric_key]


def _per_example_weights_to_per_list_weights(weights, relevance):
  """Computes per list weight from per example weight.

  Args:
    weights:  The weights `Tensor` of shape [batch_size, list_size].
    relevance:  The relevance `Tensor` of shape [batch_size, list_size].

  Returns:
    The per list `Tensor` of shape [batch_size, 1]
  """
  per_list_weights = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=weights * relevance, axis=1, keepdims=True),
      tf.reduce_sum(input_tensor=relevance, axis=1, keepdims=True))
  return per_list_weights


def _discounted_cumulative_gain(labels, weights=None):
  """Computes discounted cumulative gain (DCG).

  DCG =  SUM((2^label -1) / (log(1+rank))).

  Args:
   labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
     ideal ranking, the examples are sorted by relevance in reverse order.
    weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.

  Returns:
    A `Tensor` as the weighted discounted cumulative gain per-list. The
    tensor shape is [batch_size, 1].
  """
  list_size = tf.shape(input=labels)[1]
  position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32)
  denominator = tf.math.log(position + 1)
  numerator = tf.pow(2.0, tf.cast(labels, dtype=tf.float32)) - 1.0
  return tf.reduce_sum(
      input_tensor=weights * numerator / denominator, axis=1, keepdims=True)


def _prepare_and_validate_params(labels, predictions, weights=None, topn=None):
  """Prepares and validates the parameters.

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.

  Returns:
    (labels, predictions, weights, topn) ready to be used for metric
    calculation.
  """
  labels = tf.convert_to_tensor(value=labels)
  predictions = tf.convert_to_tensor(value=predictions)
  weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
  example_weights = tf.ones_like(labels) * weights
  predictions.get_shape().assert_is_compatible_with(example_weights.get_shape())
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  predictions.get_shape().assert_has_rank(2)
  if topn is None:
    topn = tf.shape(input=predictions)[1]

  # All labels should be >= 0. Invalid entries are reset.
  is_label_valid = utils.is_label_valid(labels)
  labels = tf.compat.v1.where(is_label_valid, labels, tf.zeros_like(labels))
  predictions = tf.compat.v1.where(
      is_label_valid, predictions, -1e-6 * tf.ones_like(predictions) +
      tf.reduce_min(input_tensor=predictions, axis=1, keepdims=True))
  return labels, predictions, example_weights, topn


class _RankingMetric(object):
  """Interface for ranking metrics."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The metric name."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def compute(self, labels, predictions, weights):
    """Computes the metric with the given inputs.

    Args:
      labels: A `Tensor` of the same shape as `predictions` representing
        relevance.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.

    Returns:
      A tf metric.
    """
    raise NotImplementedError('Calling an abstract method.')


class _MRRMetric(_RankingMetric):
  """Implements mean reciprocal rank (MRR)."""

  def __init__(self, name):
    """Constructor."""
    self._name = name

  @property
  def name(self):
    """The metric name."""
    return self._name

  def compute(self, labels, predictions, weights):
    """See `_RankingMetric`."""
    list_size = tf.shape(input=predictions)[1]
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, list_size)
    sorted_labels, = utils.sort_by_scores(predictions, [labels], topn=topn)
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    reciprocal_rank = 1.0 / tf.cast(tf.range(1, topn + 1), dtype=tf.float32)
    # MRR has a shape of [batch_size, 1].
    mrr = tf.reduce_max(
        input_tensor=relevance * reciprocal_rank, axis=1, keepdims=True)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return tf.compat.v1.metrics.mean(mrr, per_list_weights)


def mean_reciprocal_rank(labels, predictions, weights=None, name=None):
  """Computes mean reciprocal rank (MRR).

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted mean reciprocal rank of the batch.
  """
  metric = _MRRMetric(name)
  with tf.compat.v1.name_scope(metric.name, 'mean_reciprocal_rank',
                               (labels, predictions, weights)):
    return metric.compute(labels, predictions, weights)


class _ARPMetric(_RankingMetric):
  """Implements average relevance position (ARP)."""

  def __init__(self, name):
    """Constructor."""
    self._name = name

  @property
  def name(self):
    """The metric name."""
    return self._name

  def compute(self, labels, predictions, weights):
    """See `_RankingMetric`."""
    list_size = tf.shape(input=predictions)[1]
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, list_size)
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn)
    relevance = sorted_labels * sorted_weights
    position = tf.cast(tf.range(1, topn + 1), dtype=tf.float32)
    # TODO: Consider to add a cap poistion topn + 1 when there is no
    # relevant examples.
    return tf.compat.v1.metrics.mean(position * tf.ones_like(relevance),
                                     relevance)


def average_relevance_position(labels, predictions, weights=None, name=None):
  """Computes average relevance position (ARP).

  This can also be named as average_relevance_rank, but this can be confusing
  with mean_reciprocal_rank in acronyms. This name is more distinguishing and
  has been used historically for binary relevance as average_click_position.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted average relevance position.
  """
  metric = _ARPMetric(name)
  with tf.compat.v1.name_scope(metric.name, 'average_relevance_position',
                               (labels, predictions, weights)):
    return metric.compute(labels, predictions, weights)


class _PrecisionMetric(_RankingMetric):
  """Implements average relevance position (ARP)."""

  def __init__(self, name, topn):
    """Constructor."""
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def compute(self, labels, predictions, weights):
    """See `_RankingMetric`."""
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, self._topn)
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn)
    # Relevance = 1.0 when labels >= 1.0.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    per_list_precision = tf.compat.v1.math.divide_no_nan(
        tf.reduce_sum(
            input_tensor=relevance * sorted_weights, axis=1, keepdims=True),
        tf.reduce_sum(
            input_tensor=tf.ones_like(relevance) * sorted_weights,
            axis=1,
            keepdims=True))
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return tf.compat.v1.metrics.mean(per_list_precision, per_list_weights)


def precision(labels, predictions, weights=None, topn=None, name=None):
  """Computes precision as weighted average of relevant examples.

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted precision of the batch.
  """
  metric = _PrecisionMetric(name, topn)
  with tf.compat.v1.name_scope(metric.name, 'precision',
                               (labels, predictions, weights)):
    return metric.compute(labels, predictions, weights)


class _NDCGMetric(_RankingMetric):
  """Implements normalized discounted cumulative gain (NDCG)."""

  def __init__(self, name, topn):
    """Constructor."""
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def compute(self, labels, predictions, weights):
    """See `_RankingMetric`."""
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, self._topn)
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn)
    dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights)
    # Sorting over the weighted labels to get ideal ranking.
    ideal_sorted_labels, ideal_sorted_weights = utils.sort_by_scores(
        weights * labels, [labels, weights], topn=topn)
    ideal_dcg = _discounted_cumulative_gain(ideal_sorted_labels,
                                            ideal_sorted_weights)
    per_list_ndcg = tf.compat.v1.math.divide_no_nan(dcg, ideal_dcg)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.pow(2.0, tf.cast(labels, dtype=tf.float32)) - 1.0)
    return tf.compat.v1.metrics.mean(per_list_ndcg, per_list_weights)


def normalized_discounted_cumulative_gain(labels,
                                          predictions,
                                          weights=None,
                                          topn=None,
                                          name=None):
  """Computes normalized discounted cumulative gain (NDCG).

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted normalized discounted cumulative gain of the
    batch.
  """
  metric = _NDCGMetric(name, topn)
  with tf.compat.v1.name_scope(metric.name,
                               'normalized_discounted_cumulative_gain',
                               (labels, predictions, weights)):
    return metric.compute(labels, predictions, weights)


class _DCGMetric(_RankingMetric):
  """Implements discounted cumulative gain (DCG)."""

  def __init__(self, name, topn):
    """Constructor."""
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def compute(self, labels, predictions, weights):
    """See `_RankingMetric`."""
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, self._topn)
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn)
    dcg = _discounted_cumulative_gain(sorted_labels,
                                      sorted_weights) * tf.math.log1p(1.0)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.pow(2.0, tf.cast(labels, dtype=tf.float32)) - 1.0)
    return tf.compat.v1.metrics.mean(
        tf.compat.v1.math.divide_no_nan(dcg, per_list_weights),
        per_list_weights)


def discounted_cumulative_gain(labels,
                               predictions,
                               weights=None,
                               topn=None,
                               name=None):
  """Computes discounted cumulative gain (DCG).

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted discounted cumulative gain of the batch.
  """
  metric = _DCGMetric(name, topn)
  with tf.compat.v1.name_scope(name, 'discounted_cumulative_gain',
                               (labels, predictions, weights)):
    return metric.compute(labels, predictions, weights)


class _OPAMetric(_RankingMetric):
  """Implements ordered pair accuracy (OPA)."""

  def __init__(self, name):
    """Constructor."""
    self._name = name

  @property
  def name(self):
    """The metric name."""
    return self._name

  def compute(self, labels, predictions, weights):
    """See `_RankingMetric`."""
    clean_labels, predictions, weights, _ = _prepare_and_validate_params(
        labels, predictions, weights)
    label_valid = tf.equal(clean_labels, labels)
    valid_pair = tf.logical_and(
        tf.expand_dims(label_valid, 2), tf.expand_dims(label_valid, 1))
    pair_label_diff = tf.expand_dims(clean_labels, 2) - tf.expand_dims(
        clean_labels, 1)
    pair_pred_diff = tf.expand_dims(predictions, 2) - tf.expand_dims(
        predictions, 1)
    # Correct pairs are represented twice in the above pair difference tensors.
    # We only take one copy for each pair.
    correct_pairs = tf.cast(
        pair_label_diff > 0, dtype=tf.float32) * tf.cast(
            pair_pred_diff > 0, dtype=tf.float32)
    pair_weights = tf.cast(
        pair_label_diff > 0, dtype=tf.float32) * tf.expand_dims(
            weights, 2) * tf.cast(
                valid_pair, dtype=tf.float32)
    return tf.compat.v1.metrics.mean(correct_pairs, pair_weights)


def ordered_pair_accuracy(labels, predictions, weights=None, name=None):
  """Computes the percentage of correctedly ordered pair.

  For any pair of examples, we compare their orders determined by `labels` and
  `predictions`. They are correctly ordered if the two orders are compatible.
  That is, labels l_i > l_j and predictions s_i > s_j and the weight for this
  pair is the weight from the l_i.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    name: A string used as the name for this metric.

  Returns:
    A metric for the accuracy or ordered pairs.
  """
  metric = _OPAMetric(name)
  with tf.compat.v1.name_scope(metric.name, 'ordered_pair_accuracy',
                               (labels, predictions, weights)):
    return metric.compute(labels, predictions, weights)
