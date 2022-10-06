# Copyright 2022 The TensorFlow Ranking Authors.
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

  For a list with sum(relevance) = 0, we set a default weight as the following
  average weight while all the lists with sum(weights) = 0 are ignored.
    sum(per_list_weights) / num(sum(relevance) != 0 && sum(weights) != 0)
  When all the lists have sum(relevance) == 0, we set the average weight to 1.0.

  Such a computation is good for the following scenarios:
    - When all the weights are 1.0, the per list weights will be 1.0 everywhere,
      even for lists without any relevant examples because
        sum(per_list_weights) ==  num(sum(relevance) != 0)
      This handles the standard ranking metrics where the weights are all 1.0.
    - When every list has a nonzero weight, the default weight is not used. This
      handles the unbiased metrics well.
    - For the mixture of the above 2 scenario, the weights for lists with
      nonzero relevance and nonzero weights is proportional to
        per_list_weights / sum(per_list_weights) *
        num(sum(relevance) != 0) / num(lists).
      The rest have weights 1.0 / num(lists).

  Args:
    weights:  The weights `Tensor` of shape [batch_size, list_size].
    relevance:  The relevance `Tensor` of shape [batch_size, list_size].

  Returns:
    The per list `Tensor` of shape [batch_size, 1]
  """
  nonzero_weights = tf.greater(
      tf.reduce_sum(input_tensor=weights, axis=1, keepdims=True), 0.0)
  per_list_relevance = tf.reduce_sum(
      input_tensor=relevance, axis=1, keepdims=True)
  nonzero_relevance = tf.compat.v1.where(
      nonzero_weights, tf.cast(tf.greater(per_list_relevance, 0.0), tf.float32),
      tf.zeros_like(per_list_relevance))
  nonzero_relevance_count = tf.reduce_sum(
      input_tensor=nonzero_relevance, axis=0, keepdims=True)

  per_list_weights = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=weights * relevance, axis=1, keepdims=True),
      per_list_relevance)
  sum_weights = tf.reduce_sum(
      input_tensor=per_list_weights, axis=0, keepdims=True)

  avg_weight = tf.compat.v1.where(
      tf.greater(nonzero_relevance_count, 0.0),
      tf.compat.v1.math.divide_no_nan(sum_weights, nonzero_relevance_count),
      tf.ones_like(nonzero_relevance_count))
  return tf.compat.v1.where(
      nonzero_weights,
      tf.where(
          tf.greater(per_list_relevance, 0.0), per_list_weights,
          tf.ones_like(per_list_weights) * avg_weight),
      tf.zeros_like(per_list_weights))


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


def _per_list_recall(labels, predictions, topn, mask):
  """Computes the recall@k for each query in the batch.

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    topn: A cutoff for how many examples to consider for this metric.
    mask: A mask indicating which entries are valid for computing the metric.

  Returns:
    A `Tensor` of size [batch_size, 1] containing the precision of each query
    respectively.
  """
  sorted_labels = utils.sort_by_scores(predictions, [labels], topn=topn,
                                       mask=mask)[0]
  topn_positives = tf.cast(
      tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
  labels = tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32)
  per_list_recall = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=topn_positives, axis=1, keepdims=True),
      tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True))
  return per_list_recall


def _per_list_precision(labels, predictions, topn, mask):
  """Computes the precision for each query in the batch.

  Args:
    labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
      relevant example.
    predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
      the ranking score of the corresponding example.
    topn: A cutoff for how many examples to consider for this metric.
    mask: A `Tensor` of the same shape as predictions indicating which entries
      are valid for computing the metric.

  Returns:
    A `Tensor` of size [batch_size, 1] containing the precision of each query
    respectively.
  """
  sorted_labels = utils.sort_by_scores(predictions, [labels], topn=topn,
                                       mask=mask)[0]
  # Relevance = 1.0 when labels >= 1.0.
  relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
  if topn is None:
    topn = tf.shape(relevance)[1]
  valid_topn = tf.minimum(
      topn, tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=1, keepdims=True))
  per_list_precision = tf.compat.v1.math.divide_no_nan(
      tf.reduce_sum(input_tensor=relevance, axis=1, keepdims=True),
      tf.cast(valid_topn, dtype=tf.float32))
  return per_list_precision


class _RankingMetric(six.with_metaclass(abc.ABCMeta, object)):
  """Interface for ranking metrics."""

  def __init__(self, ragged=False):
    """Constructor.

    Args:
      ragged: A bool indicating whether the supplied tensors are ragged. If
        True labels, predictions and weights (if providing per-example weights)
        need to be ragged tensors with compatible shapes.
    """
    self._ragged = ragged

  @abc.abstractproperty
  def name(self):
    """The metric name."""
    raise NotImplementedError('Calling an abstract method.')

  def _prepare_and_validate_params(self, labels, predictions, weights, mask):
    """Prepares and validates the parameters.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means
        a relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      mask: A `Tensor` of the same shape as predictions indicating which entries
        are valid for computing the metric.

    Returns:
      (labels, predictions, weights, mask) ready to be used for metric
      calculation.
    """
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions, weights]):
      raise ValueError('labels, predictions and/or weights are ragged tensors, '
                       'use ragged=True to enable ragged support for metrics.')
    labels = tf.convert_to_tensor(value=labels)
    predictions = tf.convert_to_tensor(value=predictions)
    weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
    example_weights = tf.ones_like(labels) * weights
    predictions.get_shape().assert_is_compatible_with(
        example_weights.get_shape())
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions.get_shape().assert_has_rank(2)

    # All labels should be >= 0. Invalid entries are reset.
    if mask is None:
      mask = utils.is_label_valid(labels)
    mask = tf.math.logical_and(mask, tf.math.greater(example_weights, 0.0))
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    predictions = tf.compat.v1.where(
        mask, predictions, -1e-6 * tf.ones_like(predictions) +
        tf.reduce_min(input_tensor=predictions, axis=1, keepdims=True))
    return labels, predictions, example_weights, mask

  def compute(self, labels, predictions, weights=None, mask=None):
    """Computes the metric with the given inputs.

    Args:
      labels: A `Tensor` of the same shape as `predictions` representing
        relevance.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: An optional `Tensor` of the same shape of predictions or
        [batch_size, 1]. The former case is per-example and the latter case is
        per-list.
      mask: An optional `Tensor` of the same shape as predictions indicating
        which entries are valid for computing the metric. Will be ignored if
        the metric was constructed with ragged=True.

    Returns:
      A tf metric.
    """
    if self._ragged:
      labels, predictions, weights, mask = utils.ragged_to_dense(
          labels, predictions, weights)
    labels, predictions, weights, mask = self._prepare_and_validate_params(
        labels, predictions, weights, mask)
    return self._compute_impl(labels, predictions, weights, mask)

  @abc.abstractmethod
  def _compute_impl(self, labels, predictions, weights, mask):
    """Computes the metric with the given inputs.

    Args:
      labels: A `Tensor` of the same shape as `predictions` representing
        relevance.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      mask: A `Tensor` of the same shape as predictions indicating which entries
        are valid for computing the metric.

    Returns:
      A tf metric.
    """
    raise NotImplementedError('Calling an abstract method.')


class _DivRankingMetric(_RankingMetric):
  """Interface for diversity ranking metrics.

  Attributes:
    name: A string used as the name for this metric.
  """

  def __init__(self, name, topn=None, ragged=False):
    super(_DivRankingMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  @abc.abstractmethod
  def _compute_per_list_metric(self, labels, predictions, weights, topn, mask):
    """Computes the metric with the given inputs.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
        nonzero value means that the example covers the corresponding subtopic.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.
      mask: A `Tensor` of the same shape as predictions indicating which entries
        are valid for computing the metric.

    Returns:
      A tf per-list metric.
    """

  def _prepare_and_validate_params(self, labels, predictions, weights, mask):
    """Prepares and validates the parameters.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
        nonzero value means that the example covers the corresponding subtopic.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      mask: A `Tensor` of the same shape as predictions indicating which entries
        are valid for computing the metric.

    Returns:
      A 4-tuple of (labels, predictions, weights, mask) ready to be used
      for metric calculation.
    """
    labels = tf.convert_to_tensor(value=labels)
    predictions = tf.convert_to_tensor(value=predictions)
    labels.get_shape().assert_has_rank(3)
    if mask is None:
      mask = utils.is_label_valid(labels)
    mask = tf.convert_to_tensor(value=mask)
    if mask.get_shape().rank == 3:
      mask = tf.reduce_any(mask, axis=2)
    predictions = tf.where(
        mask, predictions, -1e-6 * tf.ones_like(predictions) +
        tf.reduce_min(input_tensor=predictions, axis=1, keepdims=True))
    # All labels should be >= 0. Invalid entries are reset.
    labels = tf.where(tf.expand_dims(mask, axis=2), labels,
                      tf.zeros_like(labels))
    weights = (
        tf.constant(1.0, dtype=tf.float32)
        if weights is None else tf.convert_to_tensor(value=weights))
    example_weights = tf.ones_like(predictions) * weights

    return labels, predictions, example_weights, mask

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

  def _compute_impl(self, labels, predictions, weights, mask):
    """Computes the metric and per list weight with the given inputs.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size, subtopic_size]. A
        nonzero value means that the example covers the corresponding subtopic.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      mask: An optional `Tensor` of the same shape as predictions indicating
        which entries are valid for computing the metric.

    Returns:
      A per-list metric and a per-list weights.
    """
    topn = tf.shape(input=predictions)[1] if self._topn is None else self._topn
    per_list_metric = self._compute_per_list_metric(labels, predictions,
                                                    weights, topn, mask)
    per_list_weights = self._compute_per_list_weights(weights, labels)
    return per_list_metric, per_list_weights


class MRRMetric(_RankingMetric):
  """Implements mean reciprocal rank (MRR)."""

  def __init__(self, name, topn, ragged=False):
    """Constructor."""
    super(MRRMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    sorted_labels, = utils.sort_by_scores(predictions, [labels], topn=topn,
                                          mask=mask)
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


class HitsMetric(_RankingMetric):
  r"""Implements Hits@k metric.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  Hits@k(y, s) = 1.0, if \exists i s.t. y_i >= 1 and rank(s_i) <= k
  Hits@k(y, s) = 0.0, otherwise.
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1` and `y_i = 0` if `y_i < 1`.
  NOTE: While `topn` could be left as `None` without raising an error, the Hits
  metric without `topn` specified would be trivial as it simply measures the
  percentage of lists with at least 1 relevant item.
  """

  def __init__(self, name, topn, ragged=False):
    """Constructor."""
    super(HitsMetric, self).__init__(ragged=ragged)
    self._name = name
    if topn is None:
      tf.compat.v1.logging.warning(
          'Hits metric without `topn` specified could be trivial. '
          'Consider specify `topn` for Hits metric.')
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    sorted_labels, = utils.sort_by_scores(predictions, [labels], topn=topn,
                                          mask=mask)
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    # Hits has a shape of [batch_size, 1].
    hits = tf.reduce_max(input_tensor=relevance, axis=1, keepdims=True)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return hits, per_list_weights


class ARPMetric(_RankingMetric):
  """Implements average relevance position (ARP)."""

  def __init__(self, name, ragged=False):
    """Constructor."""
    super(ARPMetric, self).__init__(ragged=ragged)
    self._name = name

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1]
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn, mask=mask)
    weighted_labels = sorted_labels * sorted_weights
    position = (tf.cast(tf.range(1, topn + 1), dtype=tf.float32) *
                tf.ones_like(weighted_labels))
    per_list_weights = tf.reduce_sum(weighted_labels, axis=1, keepdims=True)
    per_list_arp = tf.compat.v1.div_no_nan(
        tf.reduce_sum(position * weighted_labels, axis=1, keepdims=True),
        per_list_weights)
    # TODO: Consider to add a cap position topn + 1 when there is no
    # relevant examples.
    return per_list_arp, per_list_weights


class RecallMetric(_RankingMetric):
  """Implements recall@k (r@k)."""

  def __init__(self, name, topn, ragged=False):
    """Constructor."""
    super(RecallMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    per_list_recall = _per_list_recall(labels, predictions, topn, mask)
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return per_list_recall, per_list_weights


class PrecisionMetric(_RankingMetric):
  """Implements precision@k (P@k)."""

  def __init__(self, name, topn, ragged=False):
    """Constructor."""
    super(PrecisionMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    per_list_precision = _per_list_precision(labels, predictions, topn, mask)
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32))
    return per_list_precision, per_list_weights


class MeanAveragePrecisionMetric(_RankingMetric):
  """Implements mean average precision (MAP)."""

  def __init__(self, name, topn, ragged=False):
    """Constructor."""
    super(MeanAveragePrecisionMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    # Relevance = 1.0 when labels >= 1.0.
    relevance = tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32)
    sorted_relevance, sorted_weights = utils.sort_by_scores(
        predictions, [relevance, weights], topn=topn, mask=mask)
    per_list_relevant_counts = tf.cumsum(sorted_relevance, axis=1)
    per_list_cutoffs = tf.cumsum(tf.ones_like(sorted_relevance), axis=1)
    per_list_precisions = tf.math.divide_no_nan(per_list_relevant_counts,
                                                per_list_cutoffs)
    total_precision = tf.reduce_sum(
        input_tensor=per_list_precisions * sorted_weights * sorted_relevance,
        axis=1,
        keepdims=True)

    # Compute the total relevance regardless of self._topn.
    total_relevance = tf.reduce_sum(
        input_tensor=weights * relevance, axis=1, keepdims=True)

    per_list_map = tf.math.divide_no_nan(total_precision, total_relevance)
    # per_list_weights are computed from the whole list to avoid the problem of
    # 0 when there is no relevant example in topn.
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights, relevance)
    return per_list_map, per_list_weights


class NDCGMetric(_RankingMetric):
  """Implements normalized discounted cumulative gain (NDCG)."""

  def __init__(self,
               name,
               topn,
               gain_fn=_DEFAULT_GAIN_FN,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
               ragged=False):
    """Constructor."""
    super(NDCGMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn, mask=mask)
    dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights,
                                      self._gain_fn, self._rank_discount_fn)
    # Sorting over the weighted gains to get ideal ranking.
    weighted_gains = weights * self._gain_fn(tf.cast(labels, dtype=tf.float32))
    ideal_sorted_labels, ideal_sorted_weights = utils.sort_by_scores(
        weighted_gains, [labels, weights], topn=topn, mask=mask)
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
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
               ragged=False):
    """Constructor."""
    super(DCGMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn, mask=mask)
    dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights,
                                      self._gain_fn, self._rank_discount_fn)
    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=self._gain_fn(tf.cast(labels, dtype=tf.float32)))
    per_list_dcg = tf.compat.v1.math.divide_no_nan(dcg, per_list_weights)
    return per_list_dcg, per_list_weights


class OPAMetric(_RankingMetric):
  """Implements ordered pair accuracy (OPA)."""

  def __init__(self, name, ragged=False):
    """Constructor."""
    super(OPAMetric, self).__init__(ragged=ragged)
    self._name = name

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    valid_pair = tf.logical_and(
        tf.expand_dims(mask, 2), tf.expand_dims(mask, 1))
    pair_label_diff = tf.expand_dims(labels, 2) - tf.expand_dims(labels, 1)
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
    per_list_weights = tf.expand_dims(tf.reduce_sum(pair_weights, axis=[1, 2]),
                                      1)
    per_list_opa = tf.compat.v1.math.divide_no_nan(
        tf.expand_dims(tf.reduce_sum(correct_pairs * pair_weights, axis=[1, 2]),
                       1),
        per_list_weights)
    return per_list_opa, per_list_weights


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

  def _compute_per_list_metric(self, labels, predictions, weights, topn, mask):
    """See `_DivRankingMetric`."""
    sorted_labels = utils.sort_by_scores(predictions, [labels], topn=topn,
                                         mask=mask)[0]
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
    if topn is None:
      topn = tf.shape(relevance)[1]
    # valid_topn shape = [batch_size, 1].
    valid_topn = tf.minimum(topn, tf.reduce_sum(tf.cast(mask, dtype=tf.int32),
                                                axis=1, keepdims=True))
    return tf.compat.v1.math.divide_no_nan(
        tf.reduce_sum(input_tensor=relevance, axis=1, keepdims=True),
        tf.reduce_sum(
            input_tensor=tf.cast(valid_topn, dtype=tf.float32) * num_subtopics,
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
               seed=None,
               ragged=False):
    """Constructor."""
    super(AlphaDCGMetric, self).__init__(name, topn, ragged=ragged)
    self._alpha = alpha
    self._gain_fn = functools.partial(_alpha_dcg_gain_fn, alpha=alpha)
    self._rank_discount_fn = rank_discount_fn
    self._seed = seed

  def _compute_per_list_metric(self, labels, predictions, weights, topn, mask):
    """See `_DivRankingMetric`."""
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn, seed=self._seed, mask=mask)
    alpha_dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights,
                                            self._gain_fn,
                                            self._rank_discount_fn)
    per_list_weights = self._compute_per_list_weights(weights, labels)
    return tf.compat.v1.math.divide_no_nan(alpha_dcg, per_list_weights)


class BPrefMetric(_RankingMetric):
  """Implements binary preference (BPref) metric.

  In this implementation, the 0 labels are considered negative,
  any unlabelled examples should be removed or labeled as -1 prior to
  calculating BPref.
  Graded labels will be converted to binary labels by clipping to max 1.

  BPref is used in scenarios when the relevance judgements are incomplete.
  It is based on relative ranks of the judged documents and measures the
  preference for the retrieval of judged relevant documents ahead of judged
  irrelevant documents.
  The version of BPref that is used as default here was introduced in the TREC
  competition in 2005 and is described in
  https://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf :
    BPref = 1 / R SUM_r(1- |n ranked higher than r| / min(R, N))

    R = total number of relevant documents
    N = total number of irrelevant documents
    r = retrieved relevant document
    n = retrieved irrelevant document

  Note that the above trec formula is different from the other commonly cited
  version where R is used to divide |n ranked higher than r|
  instead of min(R, N):
      BPref = 1 / R SUM_r(1- |n ranked higher than r| / R)
  The potential issue of this definition is that the metric may not be monotonic
  when N > R: i.e. When a lot of irrelevant documents ranked higher than the
  relevant ones, the metric could be very positive. To use the latter formula,
  set use_trec_version to False.
  """

  def __init__(self, name, topn, use_trec_version=True, ragged=False):
    """Constructor."""
    super(BPrefMetric, self).__init__(ragged=ragged)
    self._name = name
    self._topn = topn
    self._use_trec_version = use_trec_version

  @property
  def name(self):
    """The metric name."""
    return self._name

  def _compute_impl(self, labels, predictions, weights, mask):
    """See `_RankingMetric`."""
    topn = tf.shape(predictions)[1] if self._topn is None else self._topn

    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32)
    irrelevance = tf.cast(mask, tf.float32) - relevance

    total_relevance = tf.reduce_sum(relevance, axis=1, keepdims=True)
    total_irrelevance = tf.reduce_sum(irrelevance, axis=1, keepdims=True)

    sorted_relevance, sorted_irrelevance = utils.sort_by_scores(
        predictions, [relevance, irrelevance], mask=mask, topn=topn)

    numerator = tf.minimum(
        tf.cumsum(sorted_irrelevance, axis=1), total_relevance)
    denominator = tf.minimum(
        total_irrelevance,
        total_relevance) if self._use_trec_version else total_relevance

    bpref = tf.math.divide_no_nan(
        tf.reduce_sum(((1. - tf.math.divide_no_nan(numerator, denominator)) *
                       sorted_relevance),
                      axis=1, keepdims=True), total_relevance)

    per_list_weights = _per_example_weights_to_per_list_weights(
        weights=weights,
        relevance=tf.cast(tf.greater_equal(relevance, 1.0), dtype=tf.float32))

    return bpref, per_list_weights
