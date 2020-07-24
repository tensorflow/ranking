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

"""Defines ranking metrics as TF ops.

The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tensorflow as tf

from tensorflow_ranking.python import metrics_impl
from tensorflow_ranking.python import utils

_DEFAULT_GAIN_FN = lambda label: tf.pow(2.0, label) - 1

_DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)


class RankingMetricKey(object):
  """Ranking metric key strings."""
  # Mean Reciprocal Rank. For binary relevance.
  MRR = 'mrr'

  # Average Relevance Position.
  ARP = 'arp'

  # Normalized Discounted Cumulative Gain.
  NDCG = 'ndcg'

  # Discounted Cumulative Gain.
  DCG = 'dcg'

  # Precision. For binary relevance.
  PRECISION = 'precision'

  # Mean Average Precision. For binary relevance.
  MAP = 'map'

  # PrecisionIA. For binary relevance of subtopics.
  PRECISION_IA = 'precision_ia'

  # Ordered Pair Accuracy.
  ORDERED_PAIR_ACCURACY = 'ordered_pair_accuracy'

  # Alpha Discounted Cumulative Gain.
  ALPHA_DCG = 'alpha_dcg'


def compute_mean(metric_key,
                 labels,
                 predictions,
                 weights=None,
                 topn=None,
                 name=None):
  """Returns the mean of the specified metric given the inputs.

  Args:
    metric_key: A key in `RankingMetricKey`.
    labels: A `Tensor` of the same shape as `predictions` representing
      relevance.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: An `integer` specifying the cutoff of how many items are considered in
      the metric.
    name: A `string` used as the name for this metric.

  Returns:
    A scalar as the computed metric.
  """
  metric_dict = {
      RankingMetricKey.ARP: metrics_impl.ARPMetric(metric_key),
      RankingMetricKey.MRR: metrics_impl.MRRMetric(metric_key, topn),
      RankingMetricKey.NDCG: metrics_impl.NDCGMetric(name, topn),
      RankingMetricKey.DCG: metrics_impl.DCGMetric(name, topn),
      RankingMetricKey.PRECISION: metrics_impl.PrecisionMetric(name, topn),
      RankingMetricKey.MAP: metrics_impl.MeanAveragePrecisionMetric(name, topn),
      RankingMetricKey.ORDERED_PAIR_ACCURACY: metrics_impl.OPAMetric(name),
  }
  assert metric_key in metric_dict, ('metric_key %s not supported.' %
                                     metric_key)
  metric, weight = metric_dict[metric_key].compute(labels, predictions, weights)
  return tf.compat.v1.div_no_nan(
      tf.reduce_sum(input_tensor=metric * weight),
      tf.reduce_sum(input_tensor=weight))


def make_ranking_metric_fn(metric_key,
                           weights_feature_name=None,
                           topn=None,
                           name=None,
                           gain_fn=_DEFAULT_GAIN_FN,
                           rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
                           **kwargs):
  """Factory method to create a ranking metric function.

  Args:
    metric_key: A key in `RankingMetricKey`.
    weights_feature_name: A `string` specifying the name of the weights feature
      in `features` dict.
    topn: An `integer` specifying the cutoff of how many items are considered in
      the metric.
    name: A `string` used as the name for this metric.
    gain_fn: (function) Transforms labels. A method to calculate gain parameters
      used in the definitions of the DCG and NDCG metrics, where the input is
      the relevance label of the item. The gain is often defined to be of the
      form 2^label-1.
    rank_discount_fn: (function) The rank discount function. A method to define
      the discount parameters used in the definitions of DCG and NDCG metrics,
      where the input in the rank of item. The discount function is commonly
      defined to be of the form log(rank+1).
    **kwargs: Other keyword arguments (e.g. alpha, seed).

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
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _normalized_discounted_cumulative_gain_fn(labels, predictions, features):
    """Returns normalized discounted cumulative gain as the metric."""
    return normalized_discounted_cumulative_gain(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name,
        gain_fn=gain_fn,
        rank_discount_fn=rank_discount_fn)

  def _discounted_cumulative_gain_fn(labels, predictions, features):
    """Returns discounted cumulative gain as the metric."""
    return discounted_cumulative_gain(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name,
        gain_fn=gain_fn,
        rank_discount_fn=rank_discount_fn)

  def _precision_fn(labels, predictions, features):
    """Returns precision as the metric."""
    return precision(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _mean_average_precision_fn(labels, predictions, features):
    """Returns mean average precision as the metric."""
    return mean_average_precision(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _precision_ia_fn(labels, predictions, features):
    """Returns an intent-aware precision as the metric."""
    return precision_ia(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name)

  def _ordered_pair_accuracy_fn(labels, predictions, features):
    """Returns ordered pair accuracy as the metric."""
    return ordered_pair_accuracy(
        labels, predictions, weights=_get_weights(features), name=name)

  def _alpha_discounted_cumulative_gain_fn(labels, predictions, features):
    """Returns alpha discounted cumulative gain as the metric."""
    return alpha_discounted_cumulative_gain(
        labels,
        predictions,
        weights=_get_weights(features),
        topn=topn,
        name=name,
        rank_discount_fn=rank_discount_fn,
        **kwargs)

  metric_fn_dict = {
      RankingMetricKey.ARP: _average_relevance_position_fn,
      RankingMetricKey.MRR: _mean_reciprocal_rank_fn,
      RankingMetricKey.NDCG: _normalized_discounted_cumulative_gain_fn,
      RankingMetricKey.DCG: _discounted_cumulative_gain_fn,
      RankingMetricKey.PRECISION: _precision_fn,
      RankingMetricKey.MAP: _mean_average_precision_fn,
      RankingMetricKey.PRECISION_IA: _precision_ia_fn,
      RankingMetricKey.ORDERED_PAIR_ACCURACY: _ordered_pair_accuracy_fn,
      RankingMetricKey.ALPHA_DCG: _alpha_discounted_cumulative_gain_fn,
  }
  assert metric_key in metric_fn_dict, ('metric_key %s not supported.' %
                                        metric_key)
  return metric_fn_dict[metric_key]


def mean_reciprocal_rank(labels,
                         predictions,
                         weights=None,
                         topn=None,
                         name=None):
  """Computes mean reciprocal rank (MRR).

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: An integer cutoff specifying how many examples to consider for this
      metric. If None, the whole list is considered.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted mean reciprocal rank of the batch.
  """
  metric = metrics_impl.MRRMetric(name, topn)
  with tf.compat.v1.name_scope(metric.name, 'mean_reciprocal_rank',
                               (labels, predictions, weights)):
    mrr, per_list_weights = metric.compute(labels, predictions, weights)
    return tf.compat.v1.metrics.mean(mrr, per_list_weights)


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
  metric = metrics_impl.ARPMetric(name)
  with tf.compat.v1.name_scope(metric.name, 'average_relevance_position',
                               (labels, predictions, weights)):
    arp, per_list_weights = metric.compute(labels, predictions, weights)
  return tf.compat.v1.metrics.mean(arp, per_list_weights)


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
  metric = metrics_impl.PrecisionMetric(name, topn)
  with tf.compat.v1.name_scope(metric.name, 'precision',
                               (labels, predictions, weights)):
    precision_at_k, per_list_weights = metric.compute(labels, predictions,
                                                      weights)
  return tf.compat.v1.metrics.mean(precision_at_k, per_list_weights)


def mean_average_precision(labels,
                           predictions,
                           weights=None,
                           topn=None,
                           name=None):
  """Computes mean average precision (MAP).

  The implementation of MAP is based on Equation (1.7) in the following:
  Liu, T-Y "Learning to Rank for Information Retrieval" found at
  https://www.nowpublishers.com/article/DownloadSummary/INR-016

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
    A metric for the mean average precision.
  """
  metric = metrics_impl.MeanAveragePrecisionMetric(name, topn)
  with tf.compat.v1.name_scope(metric.name, 'mean_average_precision',
                               (labels, predictions, weights)):
    per_list_map, per_list_weights = metric.compute(labels, predictions,
                                                    weights)
  return tf.compat.v1.metrics.mean(per_list_map, per_list_weights)


def precision_ia(labels, predictions, weights=None, topn=None, name=None):
  """Computes Intent-Aware Precision as weighted average of relevant examples.

  Args:
    labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
      nonzero value means that the example covers the corresponding subtopic.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.

  Returns:
    A metric for the weighted precision of the batch.
  """
  metric = metrics_impl.PrecisionIAMetric(name, topn)
  with tf.compat.v1.name_scope(metric.name, 'precision_ia',
                               (labels, predictions, weights)):
    precision_at_k, per_list_weights = metric.compute(labels, predictions,
                                                      weights)
  return tf.compat.v1.metrics.mean(precision_at_k, per_list_weights)


def normalized_discounted_cumulative_gain(
    labels,
    predictions,
    weights=None,
    topn=None,
    name=None,
    gain_fn=_DEFAULT_GAIN_FN,
    rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN):
  """Computes normalized discounted cumulative gain (NDCG).

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.
    gain_fn: (function) Transforms labels. Note that this implementation of
      NDCG assumes that this function is *increasing* as a function of its
      imput.
    rank_discount_fn: (function) The rank discount function. Note that this
      implementation of NDCG assumes that this function is *decreasing* as a
      function of its imput.

  Returns:
    A metric for the weighted normalized discounted cumulative gain of the
    batch.
  """
  metric = metrics_impl.NDCGMetric(name, topn, gain_fn, rank_discount_fn)
  with tf.compat.v1.name_scope(metric.name,
                               'normalized_discounted_cumulative_gain',
                               (labels, predictions, weights)):
    per_list_ndcg, per_list_weights = metric.compute(labels, predictions,
                                                     weights)
  return tf.compat.v1.metrics.mean(per_list_ndcg, per_list_weights)


def discounted_cumulative_gain(labels,
                               predictions,
                               weights=None,
                               topn=None,
                               name=None,
                               gain_fn=_DEFAULT_GAIN_FN,
                               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN):
  """Computes discounted cumulative gain (DCG).

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.
    gain_fn: (function) Transforms labels.
    rank_discount_fn: (function) The rank discount function.

  Returns:
    A metric for the weighted discounted cumulative gain of the batch.
  """
  metric = metrics_impl.DCGMetric(name, topn, gain_fn, rank_discount_fn)
  with tf.compat.v1.name_scope(name, 'discounted_cumulative_gain',
                               (labels, predictions, weights)):
    dcg, per_list_weights = metric.compute(labels, predictions, weights)
  return tf.compat.v1.metrics.mean(dcg, per_list_weights)


def alpha_discounted_cumulative_gain(
    labels,
    predictions,
    weights=None,
    topn=None,
    name=None,
    rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
    alpha=0.5,
    seed=None):
  """Computes alpha discounted cumulative gain (alpha-DCG).

  Args:
    labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. Each
      value represents graded relevance to a subtopic: 1 for relevent subtopic,
      0 for irrelevant, and -1 for paddings. When the actual subtopic number
      of a query is smaller than the `subtopic_size`, `labels` will be padded
      to `subtopic_size` with -1, similar to the paddings used for queries
      with doc number less then list_size.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    weights: A `Tensor` of shape [batch_size, list_size] or [batch_size, 1].
      They are per-example and per-list, respectively.
    topn: A cutoff for how many examples to consider for this metric.
    name: A string used as the name for this metric.
    rank_discount_fn: A function of rank discounts. Default is set to
      discount = 1 / log2(rank+1).
    alpha: A float between 0 and 1. Originally introduced as an assessor error
      in judging whether a document is covering a subtopic of the query. It
      can also be interpreted as the inverse number of documents covering the
      same subtopic reader needs to get and confirm the subtopic information
      of a query.
    seed: The ops-level random seed used in shuffle ties in `sort_by_scores`.

  Returns:
    A metric for the weighted alpha discounted cumulative gain of the batch.
  """
  metric = metrics_impl.AlphaDCGMetric(name, topn, alpha=alpha,
                                       rank_discount_fn=rank_discount_fn,
                                       seed=seed)
  with tf.compat.v1.name_scope(name, 'alpha_discounted_cumulative_gain',
                               (labels, predictions, weights)):
    alpha_dcg, per_list_weights = metric.compute(labels, predictions, weights)
  return tf.compat.v1.metrics.mean(alpha_dcg, per_list_weights)


def ordered_pair_accuracy(labels, predictions, weights=None, name=None):
  """Computes the percentage of correctly ordered pair.

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
  metric = metrics_impl.OPAMetric(name)
  with tf.compat.v1.name_scope(metric.name, 'ordered_pair_accuracy',
                               (labels, predictions, weights)):
    correct_pairs, pair_weights = metric.compute(labels, predictions, weights)
  return tf.compat.v1.metrics.mean(correct_pairs, pair_weights)


def eval_metric(metric_fn, **kwargs):
  """A stand-alone method to evaluate metrics on ranked results.

  Note that this method requires for the arguments of the metric to called
  explicitly. So, the correct usage is of the following form:
    tfr.metrics.eval_metric(tfr.metrics.mean_reciprocal_rank,
                            labels=my_labels,
                            predictions=my_scores).
  Here is a simple example showing how to use this method:
    import tensorflow_ranking as tfr
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    tfr.metrics.eval_metric(
        metric_fn=tfr.metrics.mean_reciprocal_rank,
        labels=labels,
        predictions=scores,
        weights=weights)
  Args:
    metric_fn: (function) Metric definition. A metric appearing in the
      TF-Ranking metrics module, e.g. tfr.metrics.mean_reciprocal_rank
    **kwargs: A collection of argument values to be passed to the metric, e.g.
      labels and predictions. See `_RankingMetric` and the various metric
      definitions in tfr.metrics for the specifics.

  Returns:
    The evaluation of the metric on the input ranked lists.

  Raises:
    ValueError: One of the arguments required by the metric is not provided in
      the list of arguments included in kwargs.

  """
  metric_spec = inspect.getargspec(metric_fn)
  metric_args = metric_spec.args
  required_metric_args = (metric_args[:-len(metric_spec.defaults)])
  for arg in required_metric_args:
    if arg not in kwargs:
      raise ValueError('Metric %s requires argument %s.' %
                       (metric_fn.__name__, arg))
  args = {}
  for arg in kwargs:
    if arg not in metric_args:
      raise ValueError('Metric %s does not accept argument %s.' %
                       (metric_fn.__name__, arg))
    args[arg] = kwargs[arg]

  with tf.compat.v1.Session() as sess:
    metric_op, update_op = metric_fn(**args)
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run([metric_op, update_op])
    return sess.run(metric_op)
