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

"""Defines ranking losses as TF ops.

The losses here are used to learn TF ranking models. It works with listwise
Tensors only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tensorflow_ranking.python import utils

# The smallest probability that is used to derive smallest logit for invalid or
# padding entries.
_EPSILON = 1e-10


class RankingLossKey(object):
  """Ranking loss key strings."""
  # Names for the ranking based loss functions.
  PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
  PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
  PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
  SOFTMAX_LOSS = 'softmax_loss'
  SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
  MEAN_SQUARED_LOSS = 'mean_squared_loss'
  LIST_MLE_LOSS = 'list_mle_loss'
  APPROX_NDCG_LOSS = 'approx_ndcg_loss'
  APPROX_MRR_LOSS = 'approx_mrr_loss'


def make_loss_fn(loss_keys,
                 loss_weights=None,
                 weights_feature_name=None,
                 lambda_weight=None,
                 reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                 name=None,
                 seed=None,
                 extra_args=None):
  """Makes a loss function using a single loss or multiple losses.

  Args:
    loss_keys: A string or list of strings representing loss keys defined in
      `RankingLossKey`. Listed loss functions will be combined in a weighted
      manner, with weights specified by `loss_weights`. If `loss_weights` is
      None, default weight of 1 will be used.
    loss_weights: List of weights, same length as `loss_keys`. Used when merging
      losses to calculate the weighted sum of losses. If `None`, all losses are
      weighted equally with weight being 1.
    weights_feature_name: A string specifying the name of the weights feature in
      `features` dict.
    lambda_weight: A `_LambdaWeight` object created by factory methods like
      `create_ndcg_lambda_weight()`.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    seed: A randomization seed used in computation of some loss functions such
      as ListMLE and pListMLE.
    extra_args: A string-keyed dictionary that contains any other loss-specific
      arguments.

  Returns:
    A function _loss_fn(). See `_loss_fn()` for its signature.

  Raises:
    ValueError: If `reduction` is invalid.
    ValueError: If `loss_keys` is None or empty.
    ValueError: If `loss_keys` and `loss_weights` have different sizes.
  """
  if (reduction not in tf.compat.v1.losses.Reduction.all() or
      reduction == tf.compat.v1.losses.Reduction.NONE):
    raise ValueError('Invalid reduction: {}'.format(reduction))

  if not loss_keys:
    raise ValueError('loss_keys cannot be None or empty.')

  if loss_weights:
    if len(loss_keys) != len(loss_weights):
      raise ValueError('loss_keys and loss_weights must have the same size.')

  if not isinstance(loss_keys, list):
    loss_keys = [loss_keys]

  def _loss_fn(labels, logits, features):
    """Computes a single loss or weighted combination of losses.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      features: Dict of Tensors of shape [batch_size, list_size, ...] for
        per-example features and shape [batch_size, ...] for non-example context
        features.

    Returns:
      An op for a single loss or weighted combination of multiple losses.

    Raises:
      ValueError: If `loss_keys` is invalid.
    """
    weights = None
    if weights_feature_name:
      weights = tf.convert_to_tensor(value=features[weights_feature_name])
      # Convert weights to a 2-D Tensor.
      weights = utils.reshape_to_2d(weights)

    loss_kwargs = {
        'labels': labels,
        'logits': logits,
        'weights': weights,
        'reduction': reduction,
        'name': name,
    }
    if extra_args is not None:
      loss_kwargs.update(extra_args)

    loss_kwargs_with_lambda_weight = loss_kwargs.copy()
    loss_kwargs_with_lambda_weight['lambda_weight'] = lambda_weight

    loss_kwargs_with_lambda_weight_and_seed = (
        loss_kwargs_with_lambda_weight.copy())
    loss_kwargs_with_lambda_weight_and_seed['seed'] = seed

    key_to_fn = {
        RankingLossKey.PAIRWISE_HINGE_LOSS:
            (_pairwise_hinge_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
            (_pairwise_logistic_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
            (_pairwise_soft_zero_one_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.SOFTMAX_LOSS:
            (_softmax_loss, loss_kwargs_with_lambda_weight),
        RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS:
            (_sigmoid_cross_entropy_loss, loss_kwargs),
        RankingLossKey.MEAN_SQUARED_LOSS: (_mean_squared_loss, loss_kwargs),
        RankingLossKey.LIST_MLE_LOSS:
            (_list_mle_loss, loss_kwargs_with_lambda_weight_and_seed),
        RankingLossKey.APPROX_NDCG_LOSS: (_approx_ndcg_loss, loss_kwargs),
        RankingLossKey.APPROX_MRR_LOSS: (_approx_mrr_loss, loss_kwargs),
    }

    # Obtain the list of loss ops.
    loss_ops = []
    for loss_key in loss_keys:
      if loss_key not in key_to_fn:
        raise ValueError('Invalid loss_key: {}.'.format(loss_key))
      loss_fn, kwargs = key_to_fn[loss_key]
      loss_ops.append(loss_fn(**kwargs))

    # Compute weighted combination of losses.
    if loss_weights:
      weighted_losses = []
      for loss_op, loss_weight in zip(loss_ops, loss_weights):
        weighted_losses.append(tf.multiply(loss_op, loss_weight))
    else:
      weighted_losses = loss_ops

    return tf.add_n(weighted_losses)

  return _loss_fn


def make_loss_metric_fn(loss_key,
                        weights_feature_name=None,
                        lambda_weight=None,
                        name=None):
  """Factory method to create a metric based on a loss.

  Args:
    loss_key: A key in `RankingLossKey`.
    weights_feature_name: A `string` specifying the name of the weights feature
      in `features` dict.
    lambda_weight: A `_LambdaWeight` object.
    name: A `string` used as the name for this metric.

  Returns:
    A metric fn with the following Args:
    * `labels`: A `Tensor` of the same shape as `predictions` representing
    graded relevance.
    * `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
    is the ranking score of the corresponding example.
    * `features`: A dict of `Tensor`s that contains all features.
  """

  metric_dict = {
      RankingLossKey.PAIRWISE_HINGE_LOSS:
          _PairwiseHingeLoss(name, reduction=None, lambda_weight=lambda_weight),
      RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
          _PairwiseLogisticLoss(
              name, reduction=None, lambda_weight=lambda_weight),
      RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
          _PairwiseSoftZeroOneLoss(
              name, reduction=None, lambda_weight=lambda_weight),
      RankingLossKey.SOFTMAX_LOSS:
          _SoftmaxLoss(name, reduction=None, lambda_weight=lambda_weight),
      RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS:
          _SigmoidCrossEntropyLoss(name, reduction=None),
      RankingLossKey.MEAN_SQUARED_LOSS:
          _MeanSquaredLoss(name, reduction=None),
      RankingLossKey.LIST_MLE_LOSS:
          _ListMLELoss(name, reduction=None, lambda_weight=lambda_weight),
      RankingLossKey.APPROX_NDCG_LOSS:
          _ApproxNDCGLoss(name, reduction=None),
      RankingLossKey.APPROX_MRR_LOSS:
          _ApproxMRRLoss(name, reduction=None),
  }

  def _get_weights(features):
    """Get weights tensor from features and reshape it to 2-D if necessary."""
    weights = None
    if weights_feature_name:
      weights = tf.convert_to_tensor(value=features[weights_feature_name])
      # Convert weights to a 2-D Tensor.
      weights = utils.reshape_to_2d(weights)
    return weights

  def metric_fn(labels, predictions, features):
    """Defines the metric fn."""
    weights = _get_weights(features)
    loss = metric_dict.get(loss_key, None)
    if loss is None:
      raise ValueError('loss_key {} not supported.'.format(loss_key))
    return loss.eval_metric(labels, predictions, weights)

  return metric_fn


def create_ndcg_lambda_weight(topn=None, smooth_fraction=0.):
  """Creates _LambdaWeight for NDCG metric."""
  return DCGLambdaWeight(
      topn,
      gain_fn=lambda labels: tf.pow(2.0, labels) - 1.,
      rank_discount_fn=lambda rank: 1. / tf.math.log1p(rank),
      normalized=True,
      smooth_fraction=smooth_fraction)


def create_reciprocal_rank_lambda_weight(topn=None, smooth_fraction=0.):
  """Creates _LambdaWeight for MRR-like metric."""
  return DCGLambdaWeight(
      topn,
      gain_fn=lambda labels: labels,
      rank_discount_fn=lambda rank: 1. / rank,
      normalized=True,
      smooth_fraction=smooth_fraction)


def create_p_list_mle_lambda_weight(list_size):
  """Creates _LambdaWeight based on Position-Aware ListMLE paper.

  Produces a weight based on the formulation presented in the
  "Position-Aware ListMLE" paper (Lan et al.) and available using
  create_p_list_mle_lambda_weight() factory function above.

  Args:
    list_size: Size of the input list.

  Returns:
    A _LambdaWeight for Position-Aware ListMLE.
  """
  return ListMLELambdaWeight(
      rank_discount_fn=lambda rank: tf.pow(2., list_size - rank) - 1.)


class _LambdaWeight(object):
  """Interface for ranking metric optimization.

  This class wraps weights used in the LambdaLoss framework for ranking metric
  optimization (https://ai.google/research/pubs/pub47258). Such an interface is
  to be instantiated by concrete lambda weight models. The instance is used
  together with standard loss such as logistic loss and softmax loss.
  """

  __metaclass__ = abc.ABCMeta

  def _get_valid_pairs_and_clean_labels(self, sorted_labels):
    """Returns a boolean Tensor for valid pairs and cleaned labels."""
    sorted_labels = tf.convert_to_tensor(value=sorted_labels)
    sorted_labels.get_shape().assert_has_rank(2)
    is_label_valid = utils.is_label_valid(sorted_labels)
    valid_pairs = tf.logical_and(
        tf.expand_dims(is_label_valid, 2), tf.expand_dims(is_label_valid, 1))
    sorted_labels = tf.compat.v1.where(is_label_valid, sorted_labels,
                                       tf.zeros_like(sorted_labels))
    return valid_pairs, sorted_labels

  @abc.abstractmethod
  def pair_weights(self, sorted_labels):
    """Returns the weight adjustment `Tensor` for example pairs.

    Args:
      sorted_labels: A dense `Tensor` of labels with shape [batch_size,
        list_size] that are sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    raise NotImplementedError('Calling an abstract method.')

  def individual_weights(self, sorted_labels):
    """Returns the weight `Tensor` for individual examples.

    Args:
      sorted_labels: A dense `Tensor` of labels with shape [batch_size,
        list_size] that are sorted by logits.

    Returns:
      A `Tensor` that can weight individual examples.
    """
    return sorted_labels


class DCGLambdaWeight(_LambdaWeight):
  """LambdaWeight for Discounted Cumulative Gain metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False,
               smooth_fraction=0.):
    """Constructor.

    Ranks are 1-based, not 0-based. Given rank i and j, there are two types of
    pair weights:
      u = |rank_discount_fn(|i-j|) - rank_discount_fn(|i-j| + 1)|
      v = |rank_discount_fn(i) - rank_discount_fn(j)|
    where u is the newly introduced one in LambdaLoss paper
    (https://ai.google/research/pubs/pub47258) and v is the original one in the
    LambdaMART paper "From RankNet to LambdaRank to LambdaMART: An Overview".
    The final pair weight contribution of ranks is
      (1-smooth_fraction) * u + smooth_fraction * v.

    Args:
      topn: (int) The topn for the DCG metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
      normalized: (bool) If True, normalize weight by the max DCG.
      smooth_fraction: (float) parameter to control the contribution from
        LambdaMART.
    """
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._normalized = normalized
    assert 0. <= smooth_fraction and smooth_fraction <= 1., (
        'smooth_fraction %s should be in range [0, 1].' % smooth_fraction)
    self._smooth_fraction = smooth_fraction

  def pair_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='dcg_lambda_weight'):
      valid_pair, sorted_labels = self._get_valid_pairs_and_clean_labels(
          sorted_labels)
      gain = self._gain_fn(sorted_labels)
      if self._normalized:
        gain *= utils.inverse_max_dcg(
            sorted_labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      pair_gain = tf.expand_dims(gain, 2) - tf.expand_dims(gain, 1)
      pair_gain *= tf.cast(valid_pair, dtype=tf.float32)

      list_size = tf.shape(input=sorted_labels)[1]
      topn = self._topn or list_size
      rank = tf.range(list_size) + 1

      def _discount_for_relative_rank_diff():
        """Rank-based discount in the LambdaLoss paper."""
        # The LambdaLoss is not well defined when topn is active and topn <
        # list_size. We cap the rank of examples to topn + 1 so that the rank
        # differene is capped to topn. This is just a convenient upperbound
        # when topn is active. We need to revisit this.
        capped_rank = tf.compat.v1.where(
            tf.greater(rank, topn),
            tf.ones_like(rank) * (topn + 1), rank)
        rank_diff = tf.cast(
            tf.abs(
                tf.expand_dims(capped_rank, 1) -
                tf.expand_dims(capped_rank, 0)),
            dtype=tf.float32)
        pair_discount = tf.compat.v1.where(
            tf.greater(rank_diff, 0),
            tf.abs(
                self._rank_discount_fn(rank_diff) -
                self._rank_discount_fn(rank_diff + 1)),
            tf.zeros_like(rank_diff))
        return pair_discount

      def _discount_for_absolute_rank():
        """Standard discount in the LambdaMART paper."""
        # When the rank discount is (1 / rank) for example, the discount is
        # |1 / r_i - 1 / r_j|. When i or j > topn, the discount becomes 0.
        rank_discount = tf.compat.v1.where(
            tf.greater(rank, topn),
            tf.zeros_like(tf.cast(rank, dtype=tf.float32)),
            self._rank_discount_fn(tf.cast(rank, dtype=tf.float32)))
        pair_discount = tf.abs(
            tf.expand_dims(rank_discount, 1) - tf.expand_dims(rank_discount, 0))
        return pair_discount

      u = _discount_for_relative_rank_diff()
      v = _discount_for_absolute_rank()
      pair_discount = (1. -
                       self._smooth_fraction) * u + self._smooth_fraction * v
      pair_weight = tf.abs(pair_gain) * pair_discount
      if self._topn is None:
        return pair_weight
      pair_mask = tf.logical_or(
          tf.expand_dims(tf.less_equal(rank, self._topn), 1),
          tf.expand_dims(tf.less_equal(rank, self._topn), 0))
      return pair_weight * tf.cast(pair_mask, dtype=tf.float32)

  def individual_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='dcg_lambda_weight'):
      sorted_labels = tf.convert_to_tensor(value=sorted_labels)
      sorted_labels = tf.compat.v1.where(
          utils.is_label_valid(sorted_labels), sorted_labels,
          tf.zeros_like(sorted_labels))
      gain = self._gain_fn(sorted_labels)
      if self._normalized:
        gain *= utils.inverse_max_dcg(
            sorted_labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      rank_discount = self._rank_discount_fn(
          tf.cast(
              tf.range(tf.shape(input=sorted_labels)[1]) + 1, dtype=tf.float32))
      return gain * rank_discount


class PrecisionLambdaWeight(_LambdaWeight):
  """LambdaWeight for Precision metric."""

  def __init__(self,
               topn,
               positive_fn=lambda label: tf.greater_equal(label, 1.0)):
    """Constructor.

    Args:
      topn: (int) The K in Precision@K metric.
      positive_fn: (function): A function on `Tensor` that output boolean True
        for positive examples. The rest are negative examples.
    """
    self._topn = topn
    self._positive_fn = positive_fn

  def pair_weights(self, sorted_labels):
    """See `_LambdaWeight`.

    The current implementation here is that for any pairs of documents i and j,
    we set the weight to be 1 if
      - i and j have different labels.
      - i <= topn and j > topn or i > topn and j <= topn.
    This is exactly the same as the original LambdaRank method. The weight is
    the gain of swapping a pair of documents.

    Args:
      sorted_labels: A dense `Tensor` of labels with shape [batch_size,
        list_size] that are sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    with tf.compat.v1.name_scope(name='precision_lambda_weight'):
      valid_pair, sorted_labels = self._get_valid_pairs_and_clean_labels(
          sorted_labels)
      binary_labels = tf.cast(
          self._positive_fn(sorted_labels), dtype=tf.float32)
      label_diff = tf.abs(
          tf.expand_dims(binary_labels, 2) - tf.expand_dims(binary_labels, 1))
      label_diff *= tf.cast(valid_pair, dtype=tf.float32)
      # i <= topn and j > topn or i > topn and j <= topn, i.e., xor(i <= topn, j
      # <= topn).
      list_size = tf.shape(input=sorted_labels)[1]
      rank = tf.range(list_size) + 1
      rank_mask = tf.math.logical_xor(
          tf.expand_dims(tf.less_equal(rank, self._topn), 1),
          tf.expand_dims(tf.less_equal(rank, self._topn), 0))
      return label_diff * tf.cast(rank_mask, dtype=tf.float32)


class ListMLELambdaWeight(_LambdaWeight):
  """LambdaWeight for ListMLE cost function."""

  def __init__(self, rank_discount_fn):
    """Constructor.

    Ranks are 1-based, not 0-based.

    Args:
      rank_discount_fn: (function) The rank discount function.
    """
    self._rank_discount_fn = rank_discount_fn

  def pair_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    return sorted_labels

  def individual_weights(self, sorted_labels):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='p_list_mle_lambda_weight'):
      sorted_labels = tf.convert_to_tensor(value=sorted_labels)
      rank_discount = self._rank_discount_fn(
          tf.cast(
              tf.range(tf.shape(input=sorted_labels)[1]) + 1, dtype=tf.float32))
      return tf.ones_like(sorted_labels) * rank_discount


def _sort_and_normalize(labels, logits, weights=None):
  """Sorts `labels` and `logits` and normalize `weights`.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1], or a `Tensor` with
      the same shape as `labels`.

  Returns:
    A tuple of (sorted_labels, sorted_logits, sorted_weights).
  """
  labels = tf.convert_to_tensor(value=labels)
  logits = tf.convert_to_tensor(value=logits)
  logits.get_shape().assert_has_rank(2)
  logits.get_shape().assert_is_compatible_with(labels.get_shape())
  weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
  weights = tf.ones_like(labels) * weights
  topn = tf.shape(input=logits)[1]

  # Only sort entries with valid labels that are >= 0.
  scores = tf.compat.v1.where(
      tf.greater_equal(labels, 0.), logits, -1e-6 * tf.ones_like(logits) +
      tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
  sorted_labels, sorted_logits, sorted_weights = utils.sort_by_scores(
      scores, [labels, logits, weights], topn=topn)
  return sorted_labels, sorted_logits, sorted_weights


def _pairwise_comparison(sorted_labels,
                         sorted_logits,
                         sorted_weights,
                         lambda_weight=None):
  r"""Returns pairwise comparison `Tensor`s.

  Given a list of n items, the labels of graded relevance l_i and the logits
  s_i, we sort the items in a list based on s_i and obtain ranks r_i. We form
  n^2 pairs of items. For each pair, we have the following:

                        /
                        | 1   if l_i > l_j
  * `pairwise_labels` = |
                        | 0   if l_i <= l_j
                        \
  * `pairwise_logits` = s_i - s_j
                         /
                         | 0              if l_i <= l_j,
  * `pairwise_weights` = | |l_i - l_j|    if lambda_weight is None,
                         | lambda_weight  otherwise.
                         \

  The `sorted_weights` is item-wise and is applied non-symmetrically to update
  pairwise_weights as
    pairwise_weights(i, j) = w_i * pairwise_weights(i, j).
  This effectively applies to all pairs with l_i > l_j. Note that it is actually
  symmetric when `sorted_weights` are constant per list, i.e., listwise weights.

  Args:
    sorted_labels: A `Tensor` with shape [batch_size, list_size] of labels
      sorted.
    sorted_logits: A `Tensor` with shape [batch_size, list_size] of logits
      sorted.
    sorted_weights: A `Tensor` with shape [batch_size, list_size] of item-wise
      weights sorted.
    lambda_weight: A `_LambdaWeight` object.

  Returns:
    A tuple of (pairwise_labels, pairwise_logits, pairwise_weights) with each
    having the shape [batch_size, list_size, list_size].
  """
  # Compute the difference for all pairs in a list. The output is a Tensor with
  # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
  # the information for pair (i, j).
  pairwise_label_diff = tf.expand_dims(sorted_labels, 2) - tf.expand_dims(
      sorted_labels, 1)
  pairwise_logits = tf.expand_dims(sorted_logits, 2) - tf.expand_dims(
      sorted_logits, 1)
  pairwise_labels = tf.cast(
      tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
  is_label_valid = utils.is_label_valid(sorted_labels)
  valid_pair = tf.logical_and(
      tf.expand_dims(is_label_valid, 2), tf.expand_dims(is_label_valid, 1))
  # Only keep the case when l_i > l_j.
  pairwise_weights = pairwise_labels * tf.cast(valid_pair, dtype=tf.float32)
  # Apply the item-wise weights along l_i.
  pairwise_weights *= tf.expand_dims(sorted_weights, 2)
  if lambda_weight is not None:
    pairwise_weights *= lambda_weight.pair_weights(sorted_labels)
  pairwise_weights = tf.stop_gradient(
      pairwise_weights, name='weights_stop_gradient')
  return pairwise_labels, pairwise_logits, pairwise_weights


class _RankingLoss(object):
  """Interface for ranking loss."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The loss name."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def compute_unreduced_loss(self, labels, logits, weights):
    """Computes the unreduced loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      A tuple of (losses, weights) before reduction.
    """
    raise NotImplementedError('Calling an abstract method.')

  def compute(self, labels, logits, weights):
    """Computes the reduced loss for training and eval.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      Reduced loss for training and eval.
    """
    losses, weights = self.compute_unreduced_loss(labels, logits, weights)
    return tf.compat.v1.losses.compute_weighted_loss(
        losses, weights, reduction=self._reduction)

  def eval_metric(self, labels, logits, weights):
    """Computes the eval metric for the loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      A metric op.
    """
    losses, weights = self.compute_unreduced_loss(labels, logits, weights)
    return tf.compat.v1.metrics.mean(losses, weights)


class _PairwiseLoss(_RankingLoss):
  """Interface for pairwise ranking loss."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, name, reduction, lambda_weight=None, params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.
      lambda_weight: A `_LambdaWeight` object.
      params: A dict for params used in loss computation.
    """
    self._name = name
    self._reduction = reduction
    self._lambda_weight = lambda_weight
    self._params = params or {}

  @property
  def name(self):
    """The loss name."""
    return self._name

  @abc.abstractmethod
  def _pairwise_loss(self, pairwise_logits):
    """The loss of pairwise logits with l_i > l_j."""
    raise NotImplementedError('Calling an abstract method.')

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    sorted_labels, sorted_logits, sorted_weights = _sort_and_normalize(
        labels, logits, weights)
    _, pairwise_logits, pairwise_weights = _pairwise_comparison(
        sorted_labels, sorted_logits, sorted_weights, self._lambda_weight)
    if self._lambda_weight is not None:
      # For LambdaLoss with relative rank difference, the scale of loss becomes
      # much smaller when applying LambdaWeight. This affects the training can
      # make the optimal learning rate become much larger. We use a heuristic to
      # scale it up to the same magnitude as standard pairwise loss.
      pairwise_weights *= tf.cast(
          tf.shape(input=sorted_labels)[1], dtype=tf.float32)
    return self._pairwise_loss(pairwise_logits), pairwise_weights


class _PairwiseLogisticLoss(_PairwiseLoss):
  """Implements pairwise logistic loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    # The following is the same as log(1 + exp(-pairwise_logits)).
    return tf.nn.relu(-pairwise_logits) + tf.math.log1p(
        tf.exp(-tf.abs(pairwise_logits)))


class _PairwiseHingeLoss(_PairwiseLoss):
  """Implements pairwise hinge loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    return tf.nn.relu(1 - pairwise_logits)


class _PairwiseSoftZeroOneLoss(_PairwiseLoss):
  """Implements pairwise hinge loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    return tf.compat.v1.where(
        tf.greater(pairwise_logits, 0), 1. - tf.sigmoid(pairwise_logits),
        tf.sigmoid(-pairwise_logits))


def _pairwise_hinge_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the pairwise hinge loss for a list.

  The hinge loss is defined as Hinge(l_i > l_j) = max(0, 1 - (s_i - s_j)). So a
  correctly ordered pair has 0 loss if (s_i - s_j >= 1). Otherwise the loss
  increases linearly with s_i - s_j. When the list_size is 2, this reduces to
  the standard hinge loss.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the pairwise hinge loss.
  """
  loss = _PairwiseHingeLoss(name, reduction, lambda_weight)
  with tf.compat.v1.name_scope(loss.name, 'pairwise_hinge_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


def _pairwise_logistic_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the pairwise logistic loss for a list.

  The preference probability of each pair is computed as the sigmoid function:
  P(l_i > l_j) = 1 / (1 + exp(s_j - s_i)) and the logistic loss is log(P(l_i >
  l_j)) if l_i > l_j and 0 otherwise.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the pairwise logistic loss.
  """
  loss = _PairwiseLogisticLoss(name, reduction, lambda_weight)
  with tf.compat.v1.name_scope(loss.name, 'pairwise_logistic_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


def _pairwise_soft_zero_one_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the pairwise soft zero-one loss.

  Note this is different from sigmoid cross entropy in that soft zero-one loss
  is a smooth but non-convex approximation of zero-one loss. The preference
  probability of each pair is computed as the sigmoid function: P(l_i > l_j) = 1
  / (1 + exp(s_j - s_i)). Then 1 - P(l_i > l_j) is directly used as the loss.
  So a correctly ordered pair has a loss close to 0, while an incorrectly
  ordered pair has a loss bounded by 1.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the pairwise soft zero one loss.
  """
  loss = _PairwiseSoftZeroOneLoss(name, reduction, lambda_weight)
  with tf.compat.v1.name_scope(loss.name, 'pairwise_soft_zero_one_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


class _ListwiseLoss(_RankingLoss):
  """Interface for listwise loss."""

  def __init__(self,
               name,
               reduction,
               lambda_weight=None,
               seed=None,
               params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.
      lambda_weight: A `_LambdaWeight` object.
      seed: A randomization seed used when shuffling ground truth permutations.
      params: A dict for params used in loss computation.
    """
    self._name = name
    self._reduction = reduction
    self._lambda_weight = lambda_weight
    self._seed = seed
    self._params = params or {}

  @property
  def name(self):
    """The loss name."""
    return self._name


class _SoftmaxLoss(_ListwiseLoss):
  """Implements softmax loss."""

  def _precompute(self, labels, logits, weights):
    """Precomputes Tensors for softmax cross entropy inputs."""
    sorted_labels, sorted_logits, sorted_weights = _sort_and_normalize(
        labels, logits, weights)
    is_label_valid = utils.is_label_valid(sorted_labels)
    # Reset the invalid labels to 0 and reset the invalid logits to a logit with
    # ~= 0 contribution in softmax.
    sorted_labels = tf.compat.v1.where(is_label_valid, sorted_labels,
                                       tf.zeros_like(sorted_labels))
    sorted_logits = tf.compat.v1.where(
        is_label_valid, sorted_logits,
        tf.math.log(_EPSILON) * tf.ones_like(sorted_logits))
    if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                      DCGLambdaWeight):
      sorted_labels = self._lambda_weight.individual_weights(sorted_labels)
    sorted_labels *= sorted_weights
    label_sum = tf.reduce_sum(input_tensor=sorted_labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    padded_sorted_labels = tf.compat.v1.where(
        nonzero_mask, sorted_labels, _EPSILON * tf.ones_like(sorted_labels))
    padded_label_sum = tf.reduce_sum(
        input_tensor=padded_sorted_labels, axis=1, keepdims=True)
    labels_for_softmax = padded_sorted_labels / padded_label_sum
    logits_for_softmax = sorted_logits
    # Padded labels have 0 weights in label_sum.
    weights_for_softmax = tf.reshape(label_sum, [-1])
    return labels_for_softmax, logits_for_softmax, weights_for_softmax

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    labels, logits, weights = self._precompute(labels, logits, weights)
    losses = tf.compat.v1.losses.softmax_cross_entropy(
        labels, logits, reduction=tf.compat.v1.losses.Reduction.NONE)
    return losses, weights

  def compute(self, labels, logits, weights):
    """See `_RankingLoss`."""
    labels, logits, weights = self._precompute(labels, logits, weights)
    return tf.compat.v1.losses.softmax_cross_entropy(
        labels, logits, weights=weights, reduction=self._reduction)


def _softmax_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the softmax cross entropy for a list.

  This is the ListNet loss originally proposed by Cao et al.
  ["Learning to Rank: From Pairwise Approach to Listwise Approach"] and is
  appropriate for datasets with binary relevance labels [see "An Analysis of
  the Softmax Cross Entropy Loss for Learning-to-Rank with Binary Relevance" by
  Bruch et al.]

  Given the labels l_i and the logits s_i, we sort the examples and obtain ranks
  r_i. The standard softmax loss doesn't need r_i and is defined as
      -sum_i l_i * log(exp(s_i) / (exp(s_1) + ... + exp(s_n))).
  The `lambda_weight` re-weight examples based on l_i and r_i.
      -sum_i w(l_i, r_i) * log(exp(s_i) / (exp(s_1) + ... + exp(s_n))).abc
  See 'individual_weights' in 'DCGLambdaWeight' for how w(l_i, r_i) is computed.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `DCGLambdaWeight` instance.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the softmax cross entropy as a loss.
  """
  loss = _SoftmaxLoss(name, reduction, lambda_weight)
  with tf.compat.v1.name_scope(loss.name, 'softmax_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


class _PointwiseLoss(_RankingLoss):
  """Interface for pointwise loss."""

  def __init__(self, name, reduction, params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.
      params: A dict for params used in loss computation.
    """
    self._name = name
    self._reduction = reduction
    self._params = params or {}

  @property
  def name(self):
    """The loss name."""
    return self._name


class _SigmoidCrossEntropyLoss(_PointwiseLoss):
  """Implements sigmoid cross entropy loss."""

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
    weights = tf.compat.v1.where(
        utils.is_label_valid(labels),
        tf.ones_like(labels) * weights, tf.zeros_like(labels))
    losses = tf.compat.v1.losses.sigmoid_cross_entropy(
        labels, logits, reduction=tf.compat.v1.losses.Reduction.NONE)
    return losses, weights


def _sigmoid_cross_entropy_loss(
    labels,
    logits,
    weights=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the sigmoid_cross_entropy loss for a list.

  Given the labels of graded relevance l_i and the logits s_i, we calculate
  the sigmoid cross entropy for each ith position and aggregate the per position
  losses.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the sigmoid cross entropy as a loss.
  """
  loss = _SigmoidCrossEntropyLoss(name, reduction)
  with tf.compat.v1.name_scope(loss.name, 'sigmoid_cross_entropy_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


class _MeanSquaredLoss(_PointwiseLoss):
  """Implements the means squared error loss."""

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
    weights = tf.compat.v1.where(
        utils.is_label_valid(labels),
        tf.ones_like(labels) * weights, tf.zeros_like(labels))
    losses = tf.compat.v1.losses.mean_squared_error(
        labels, logits, reduction=tf.compat.v1.losses.Reduction.NONE)
    return losses, weights


def _mean_squared_loss(
    labels,
    logits,
    weights=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None):
  """Computes the mean squared loss for a list.

  Given the labels of graded relevance l_i and the logits s_i, we calculate
  the squared error for each ith position and aggregate the per position
  losses.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

  Returns:
    An op for the mean squared error as a loss.
  """
  loss = _MeanSquaredLoss(name, reduction)
  with tf.compat.v1.name_scope(loss.name, 'mean_squared_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


class _ListMLELoss(_ListwiseLoss):
  """Implements ListMLE loss."""

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    is_label_valid = utils.is_label_valid(labels)
    # Reset the invalid labels to 0 and reset the invalid logits to a logit with
    # ~= 0 contribution.
    labels = tf.compat.v1.where(is_label_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(is_label_valid, logits,
                                tf.math.log(_EPSILON) * tf.ones_like(logits))
    weights = 1.0 if weights is None else tf.convert_to_tensor(value=weights)
    weights = tf.squeeze(weights)

    # Shuffle labels and logits to add randomness to sort.
    shuffled_indices = utils.shuffle_valid_indices(is_label_valid, self._seed)
    shuffled_labels = tf.gather_nd(labels, shuffled_indices)
    shuffled_logits = tf.gather_nd(logits, shuffled_indices)

    # TODO: Remove the shuffling above and use
    # shuffle_ties=True.
    sorted_labels, sorted_logits = utils.sort_by_scores(
        shuffled_labels, [shuffled_labels, shuffled_logits], shuffle_ties=False)

    raw_max = tf.reduce_max(input_tensor=sorted_logits, axis=1, keepdims=True)
    sorted_logits = sorted_logits - raw_max
    sums = tf.cumsum(tf.exp(sorted_logits), axis=1, reverse=True)
    sums = tf.math.log(sums) - sorted_logits

    if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                      ListMLELambdaWeight):
      sums *= self._lambda_weight.individual_weights(sorted_labels)

    negative_log_likelihood = tf.reduce_sum(input_tensor=sums, axis=1)
    return negative_log_likelihood, weights


def _list_mle_loss(
    labels,
    logits,
    weights=None,
    lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None,
    seed=None):
  """Computes the ListMLE loss [Xia et al.

  2008] for a list.

  Given the labels of graded relevance l_i and the logits s_i, we calculate
  the ListMLE loss for the given list.

  The `lambda_weight` re-weights examples based on l_i and r_i.
  The recommended weighting scheme is the formulation presented in the
  "Position-Aware ListMLE" paper (Lan et al.) and available using
  create_p_list_mle_lambda_weight() factory function above.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `DCGLambdaWeight` instance.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    seed: A randomization seed used when shuffling ground truth permutations.

  Returns:
    An op for the ListMLE loss.
  """
  loss = _ListMLELoss(name, reduction, lambda_weight, seed)
  with tf.compat.v1.name_scope(loss.name, 'list_mle_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


class _ApproxNDCGLoss(_ListwiseLoss):
  """Implements ApproxNDCG loss."""

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    alpha = self._params.get('alpha', 10.0)
    is_label_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_label_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_label_valid, logits, -1e3 * tf.ones_like(logits) +
        tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    if weights is None:
      weights = tf.ones_like(label_sum)
    weights = tf.squeeze(weights)

    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    weights = tf.compat.v1.where(nonzero_mask, weights, tf.zeros_like(weights))

    gains = tf.pow(2., tf.cast(labels, dtype=tf.float32)) - 1.
    ranks = utils.approx_ranks(logits, alpha=alpha)
    discounts = 1. / tf.math.log1p(ranks)
    dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1)
    cost = -dcg * tf.squeeze(utils.inverse_max_dcg(labels))
    return cost, weights


def _approx_ndcg_loss(labels,
                      logits,
                      weights=None,
                      reduction=tf.compat.v1.losses.Reduction.SUM,
                      name=None,
                      alpha=10.):
  """Computes ApproxNDCG loss.

  ApproxNDCG ["A general approximation framework for direct optimization of
  information retrieval measures" by Qin et al.] is a smooth approximation
  to NDCG. Its performance on datasets with graded relevance is competitive
  with other state-of-the-art algorithms [see "Revisiting Approximate Metric
  Optimization in the Age of Deep Neural Networks" by Bruch et al.].

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights. If None, the weight of a list in the mini-batch is set to the sum
      of the labels of the items in that list.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    alpha: The exponent in the generalized sigmoid function.

  Returns:
    An op for the ApproxNDCG loss.
  """
  loss = _ApproxNDCGLoss(name, reduction, params={'alpha': alpha})
  with tf.compat.v1.name_scope(loss.name, 'approx_ndcg_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)


class _ApproxMRRLoss(_ListwiseLoss):
  """Implements ApproxMRR loss."""

  def compute_unreduced_loss(self, labels, logits, weights):
    """See `_RankingLoss`."""
    alpha = self._params.get('alpha', 10.0)
    is_label_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_label_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_label_valid, logits, -1e3 * tf.ones_like(logits) +
        tf.math.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.math.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    if weights is None:
      weights = tf.ones_like(label_sum)
    weights = tf.squeeze(weights)

    nonzero_mask = tf.math.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    weights = tf.compat.v1.where(nonzero_mask, weights, tf.zeros_like(weights))

    rr = 1. / utils.approx_ranks(logits, alpha=alpha)
    rr = tf.math.reduce_sum(input_tensor=rr * labels, axis=-1)
    mrr = rr / tf.math.reduce_sum(input_tensor=labels, axis=-1)
    cost = -mrr
    return cost, weights


def _approx_mrr_loss(labels,
                     logits,
                     weights=None,
                     reduction=tf.compat.v1.losses.Reduction.SUM,
                     name=None,
                     alpha=10.):
  """Computes ApproxMRR loss.

  ApproxMRR ["A general approximation framework for direct optimization of
  information retrieval measures" by Qin et al.] is a smooth approximation
  to MRR.

  Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights. If None, the weight of a list in the mini-batch is set to the sum
      of the labels of the items in that list.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    alpha: The exponent in the generalized sigmoid function.

  Returns:
    An op for the ApproxMRR loss.
  """
  loss = _ApproxMRRLoss(name, reduction, params={'alpha': alpha})
  with tf.compat.v1.name_scope(loss.name, 'approx_mrr_loss',
                               (labels, logits, weights)):
    return loss.compute(labels, logits, weights)
