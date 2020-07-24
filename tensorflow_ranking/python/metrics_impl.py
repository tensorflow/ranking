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
"""Implements the metrics for TF-Ranking.

The test cases are mainly on metrics_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import six
import tensorflow as tf

from tensorflow_ranking.python import utils

_DEFAULT_GAIN_FN = lambda label: tf.pow(2.0, label) - 1

_DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)


def _alpha_dcg_gain_fn(labels, alpha):
  """Computes gain for alpha DCG metric from sorted labels.

  Args:
    labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. Each
      value represents graded relevance to a subtopic: 1 for relevent subtopic,
      0 for irrelevant, and -1 for paddings. When the actual subtopic number of
      a query is smaller than the `subtopic_size`, `labels` will be padded to
      `subtopic_size` with -1, similar to the paddings used for queries with doc
      number less then list_size.
    alpha: A float between 0 and 1. Originally introduced as an assessor error
      in judging whether a document is covering a subtopic of the query. It can
      also be interpreted as the inverse number of documents covering the same
      subtopic reader needs to get and confirm the subtopic information of a
      query.

  Returns:
    A function computes the alpha DCG gain.
  """
  # Cumulative number of topics covered along the list_size dimension.
  cum_subtopics = tf.cumsum(labels, axis=1, exclusive=True)
  gains = tf.reduce_sum(
      tf.multiply(labels, tf.pow(1 - alpha, cum_subtopics)), axis=-1)

  return gains


def _per_example_weights_to_per_list_weights(weights, relevance):
  """Computes per list weight from per example weight.

  The per-list weights are computed as:
    per_list_weights = sum(weights * relevance) / sum(relevance).

  For the list with sum(relevance) = 0, we set a default weight as the following
  average weight:
    sum(per_list_weights) / num(sum(relevance) != 0)

  Such a computation is good for the following scenarios:
    - When all the weights are 1.0, the per list weights will be 1.0 everywhere,
      even for lists without any relevant examples because
        sum(per_list_weights) ==  num(sum(relevance) != 0)
      This handles the standard ranking metrics where the weights are all 1.0.
    - When every list has a nonzero weight, the default weight is not used. This
      handles the unbiased metrics well.
    - For the mixture of the above 2 scenario, the weights for lists with
      nonzero relevance is proportional to
        per_list_weights / sum(per_list_weights) *
        num(sum(relevance) != 0) / num(lists).
      The rest have weights 1.0 / num(lists).

  Args:
    weights:  The weights `Tensor` of shape [batch_size, list_size].
    relevance:  The relevance `Tensor` of shape [batch_size, list_size].

  Returns:
    The per list `Tensor` of shape [batch_size, 1]
  """
  per_list_relevance = tf.reduce_sum(
      input_tensor=relevance, axis=1, keepdims=True)
  nonzero_relevance = tf.cast(tf.greater(per_list_relevance, 0.0), tf.float32)
  nonzero_relevance_count = tf.reduce_sum(
      input_tensor=nonzero_relevance, axis=0, keepdims=True)

  per_list_weights = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=weights * relevance, axis=1, keepdims=True),
      per_list_relevance)
  sum_weights = tf.reduce_sum(
      input_tensor=per_list_weights, axis=0, keepdims=True)

  avg_weight = tf.compat.v1.math.divide_no_nan(sum_weights,
                                               nonzero_relevance_count)
  return tf.compat.v1.where(
      tf.greater(per_list_relevance, 0.0), per_list_weights,
      tf.ones_like(per_list_weights) * avg_weight)


def _discounted_cumulative_gain(labels,
                                weights=None,
                                gain_fn=_DEFAULT_GAIN_FN,
                                rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN):
  """Computes discounted cumulative gain (DCG).

  DCG = SUM(gain_fn(label) / rank_discount_fn(rank)). Using the default values
  of the gain and discount functions, we get the following commonly used
  formula for DCG: SUM((2^label -1) / log(1+rank)).

  Args:
    labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
      ideal ranking, the examples are sorted by relevance in reverse order. In
      alpha_dcg, it is a `Tensor` with shape [batch_size, list_size,
      subtopic_size].
    weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    gain_fn: (function) Transforms labels.
    rank_discount_fn: (function) The rank discount function.

  Returns:
    A `Tensor` as the weighted discounted cumulative gain per-list. The
    tensor shape is [batch_size, 1].
  """
  list_size = tf.shape(input=labels)[1]
  position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32)
  gain = gain_fn(tf.cast(labels, dtype=tf.float32))
  discount = rank_discount_fn(position)
  return tf.reduce_sum(
      input_tensor=weights * gain * discount, axis=1, keepdims=True)


def _per_list_precision(labels, predictions, topn):
  """Computes the precision for each query in the batch.

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    topn: A cutoff for how many examples to consider for this metric.

  Returns:
    A `Tensor` of size [batch_size, 1] containing the precision of each query
    respectively.
  """
  sorted_labels = utils.sort_by_scores(predictions, [labels], topn=topn)[0]
  # Relevance = 1.0 when labels >= 1.0.
  relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
  per_list_precision = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=relevance, axis=1, keepdims=True),
      tf.reduce_sum(
          input_tensor=tf.ones_like(relevance), axis=1, keepdims=True))
  return per_list_precision


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


class _RankingMetric(six.with_metaclass(abc.ABCMeta, object)):
  """Interface for ranking metrics."""

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


class _DivRankingMetric(_RankingMetric):
  """Interface for diversity ranking metrics.

  Attributes:
    name: A string used as the name for this metric.
  """

  def __init__(self, name, topn=None):
    super(_DivRankingMetric, self).__init__()
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  @abc.abstractmethod
  def _compute_per_list_metric(self, labels, predictions, weights, topn):
    """Computes the metric with the given inputs.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
        nonzero value means that the example covers the corresponding subtopic.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.

    Returns:
      A tf per-list metric.
    """

  def _prepare_and_validate_params(self,
                                   labels,
                                   predictions,
                                   weights=None):
    """Prepares and validates the parameters.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
        nonzero value means that the example covers the corresponding subtopic.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.

    Returns:
      A 4-tuple of (labels, predictions, weights, topn) ready to be used for
      metric calculation.
    """
    labels = tf.convert_to_tensor(value=labels)
    predictions = tf.convert_to_tensor(value=predictions)
    labels.get_shape().assert_has_rank(3)

    is_label_valid = utils.is_label_valid(labels)
    predictions = tf.where(
        tf.reduce_any(is_label_valid,
                      axis=-1), predictions, -1e-6 * tf.ones_like(predictions) +
        tf.reduce_min(input_tensor=predictions, axis=1, keepdims=True))
    # All labels should be >= 0. Invalid entries are reset.
    labels = tf.where(is_label_valid, labels, tf.zeros_like(labels))
    weights = (
        tf.constant(1.0, dtype=tf.float32)
        if weights is None else tf.convert_to_tensor(value=weights))
    example_weights = tf.ones_like(predictions) * weights
    topn = tf.shape(input=predictions)[1] if self._topn is None else self._topn

    return labels, predictions, example_weights, topn

  def _compute_per_list_weights(self, weights, labels):
    """Computes per list weight from weights and labels for diversification.

    Args:
      weights:  The weights `Tensor` of shape [batch_size, list_size].
      labels:  The labels `Tensor` of shape [batch_size, list_size,
        subtopic_size].

    Returns:
      The per-list `Tensor` of shape [batch_size, 1]
    """
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    return _per_example_weights_to_per_list_weights(
        weights,
        tf.cast(
            tf.reduce_any(tf.greater_equal(labels, 1.0), axis=-1),
            dtype=tf.float32))

  def compute(self, labels, predictions, weights):
    """Computes the metric and per list weight with the given inputs.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
        nonzero value means that the example covers the corresponding subtopic.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.

    Returns:
      A per-list metric and a per-list weights.
    """
    labels, predictions, weights, topn = (
        self._prepare_and_validate_params(labels, predictions, weights))
    per_list_metric = self._compute_per_list_metric(
        labels, predictions, weights, topn)
    per_list_weights = self._compute_per_list_weights(weights, labels)
    return per_list_metric, per_list_weights


class MRRMetric(_RankingMetric):
  """Implements mean reciprocal rank (MRR)."""

  def __init__(self, name, topn):
    """Constructor."""
    super(MRRMetric, self).__init__()
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
    sorted_labels, = utils.sort_by_scores(predictions, [labels], topn=topn)
    sorted_list_size = tf.shape(input=sorted_labels)[1]
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    reciprocal_rank = 1.0 / tf.cast(
        tf.range(1, sorted_list_size + 1), dtype=tf.float32)
    # MRR has a shape of [batch_size, 1].
    mrr = tf.reduce_max(
        input_tensor=relevance * reciprocal_rank, axis=1, keepdims=True)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return mrr, per_list_weights


class ARPMetric(_RankingMetric):
  """Implements average relevance position (ARP)."""

  def __init__(self, name):
    """Constructor."""
    super(ARPMetric, self).__init__()
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
    # TODO: Consider to add a cap position topn + 1 when there is no
    # relevant examples.
    return position * tf.ones_like(relevance), relevance


class PrecisionMetric(_RankingMetric):
  """Implements precision@k (P@k)."""

  def __init__(self, name, topn):
    """Constructor."""
    super(PrecisionMetric, self).__init__()
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
    per_list_precision = _per_list_precision(labels, predictions, topn)
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return per_list_precision, per_list_weights


class MeanAveragePrecisionMetric(_RankingMetric):
  """Implements mean average precision (MAP)."""

  def __init__(self, name, topn):
    """Constructor."""
    super(MeanAveragePrecisionMetric, self).__init__()
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
    sorted_relevance = tf.cast(
        tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    per_list_relevant_counts = tf.cumsum(sorted_relevance, axis=1)
    per_list_cutoffs = tf.cumsum(tf.ones_like(sorted_relevance), axis=1)
    per_list_precisions = tf.math.divide_no_nan(per_list_relevant_counts,
                                                per_list_cutoffs)
    total_precision = tf.reduce_sum(
        input_tensor=per_list_precisions * sorted_weights * sorted_relevance,
        axis=1,
        keepdims=True)
    total_relevance = tf.reduce_sum(
        input_tensor=sorted_weights * sorted_relevance, axis=1, keepdims=True)
    per_list_map = tf.math.divide_no_nan(total_precision, total_relevance)
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return per_list_map, per_list_weights


class NDCGMetric(_RankingMetric):
  """Implements normalized discounted cumulative gain (NDCG)."""

  def __init__(self,
               name,
               topn,
               gain_fn=_DEFAULT_GAIN_FN,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN):
    """Constructor."""
    super(NDCGMetric, self).__init__()
    self._name = name
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn

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
    dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights,
                                      self._gain_fn, self._rank_discount_fn)
    # Sorting over the weighted labels to get ideal ranking.
    ideal_sorted_labels, ideal_sorted_weights = utils.sort_by_scores(
        weights * labels, [labels, weights], topn=topn)
    ideal_dcg = _discounted_cumulative_gain(ideal_sorted_labels,
                                            ideal_sorted_weights, self._gain_fn,
                                            self._rank_discount_fn)
    per_list_ndcg = tf.compat.v1.math.divide_no_nan(dcg, ideal_dcg)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=self._gain_fn(tf.cast(labels, dtype=tf.float32)))
    return per_list_ndcg, per_list_weights


class DCGMetric(_RankingMetric):
  """Implements discounted cumulative gain (DCG)."""

  def __init__(self,
               name,
               topn,
               gain_fn=_DEFAULT_GAIN_FN,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN):
    """Constructor."""
    super(DCGMetric, self).__init__()
    self._name = name
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn

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
    dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights,
                                      self._gain_fn, self._rank_discount_fn)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=self._gain_fn(tf.cast(labels, dtype=tf.float32)))
    per_list_dcg = tf.compat.v1.math.divide_no_nan(dcg, per_list_weights)
    return per_list_dcg, per_list_weights


class OPAMetric(_RankingMetric):
  """Implements ordered pair accuracy (OPA)."""

  def __init__(self, name):
    """Constructor."""
    super(OPAMetric, self).__init__()
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
    return correct_pairs, pair_weights


class PrecisionIAMetric(_DivRankingMetric):
  """Implements Intent-Aware Precision@k (Pre-IA@k).

  PrecisionIA is a metric introduced in ["Overview of the TREC 2009 Web Track."]
  by C Clarke, et al. It is one of the evaluation measures for the TREC
  diversity task, where a query may have multiple different implications, termed
  as subtopics / nuggets. Specifically,
    Pre-IA@k = SUM_t SUM_{i=1}^k label(rank=i, topic=t) / (# of Subtopics * k),
  where t indexes subtopics and i indexes document ranks, SUM_t sums over all
  subtopics and SUM_{i=1}^k sums over the top k ranks.
  """

  def _compute_per_list_metric(self, labels, predictions, weights, topn):
    """See `_DivRankingMetric`."""
    sorted_labels = utils.sort_by_scores(predictions, [labels], topn=topn)[0]
    # relevance shape = [batch_size, topn].
    relevance = tf.reduce_sum(
        tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32),
        axis=-1)
    # num_subtopics shape = [batch_size, 1].
    num_subtopics = tf.reduce_sum(
        tf.cast(
            tf.reduce_any(tf.greater_equal(labels, 1.0), axis=1, keepdims=True),
            dtype=tf.float32),
        axis=-1)
    return tf.compat.v1.math.divide_no_nan(
        tf.reduce_sum(input_tensor=relevance, axis=1, keepdims=True),
        tf.reduce_sum(
            input_tensor=tf.ones_like(relevance) * num_subtopics,
            axis=1,
            keepdims=True))


class AlphaDCGMetric(_DivRankingMetric):
  """Implements alpha discounted cumulative gain (alphaDCG).

  alphaDCG is a metric first introduced in ["Novelty and Diversity in
  Information Retrieval Evaluation."] by C Clarke, et al. It is commonly used in
  diversification tasks, where a query may have multiple different implications,
  termed as subtopics / nuggets. This metric tends to emphasize a rank with
  items covering different subtopics on top by a gain_fn with reduced gain from
  readily covered subtopics. Specifically,
    alphaDCG = SUM(gain_fn(label, alpha) / rank_discount_fn(rank)).
  Using the default values of the gain and discount functions, we get the
  following commonly used formula for alphaDCG:
    SUM(label_i * (1-alpha)^(SUM_{rank_j<rank_i}label_j) / log2(1+rank_i)).
  """

  def __init__(self,
               name,
               topn,
               alpha=0.5,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
               seed=None):
    """Constructor."""
    super(AlphaDCGMetric, self).__init__(name, topn)
    self._alpha = alpha
    self._gain_fn = functools.partial(_alpha_dcg_gain_fn, alpha=alpha)
    self._rank_discount_fn = rank_discount_fn
    self._seed = seed

  def _compute_per_list_metric(self, labels, predictions, weights, topn):
    """See `_DivRankingMetric`."""
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn, seed=self._seed)
    alpha_dcg = _discounted_cumulative_gain(sorted_labels,
                                            sorted_weights,
                                            self._gain_fn,
                                            self._rank_discount_fn)
    per_list_weights = self._compute_per_list_weights(weights, labels)
    return tf.compat.v1.math.divide_no_nan(alpha_dcg, per_list_weights)
