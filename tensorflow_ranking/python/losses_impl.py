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

"""Implements the losses for TF-Ranking.

The test cases are mainly on losses_test.py.
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


def _check_tensor_shapes(tensors):
  """Checks the tensor shapes to be compatible."""
  for tensor in tensors:
    tensor = tf.convert_to_tensor(value=tensor)
    tensor.get_shape().assert_has_rank(2)
    tensor.get_shape().assert_is_compatible_with(
        tf.convert_to_tensor(value=tensors[0]).get_shape())


def _apply_pairwise_op(op, tensor):
  """Applies the op on tensor in the pairwise manner."""
  _check_tensor_shapes([tensor])
  return op(tf.expand_dims(tensor, 2), tf.expand_dims(tensor, 1))


def _get_valid_pairs_and_clean_labels(labels):
  """Returns a boolean Tensor for valid pairs and cleaned labels."""
  labels = tf.convert_to_tensor(value=labels)
  labels.get_shape().assert_has_rank(2)
  is_valid = utils.is_label_valid(labels)
  valid_pairs = _apply_pairwise_op(tf.logical_and, is_valid)
  labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
  return valid_pairs, labels


class _LambdaWeight(object):
  """Interface for ranking metric optimization.

  This class wraps weights used in the LambdaLoss framework for ranking metric
  optimization (https://ai.google/research/pubs/pub47258). Such an interface is
  to be instantiated by concrete lambda weight models. The instance is used
  together with standard loss such as logistic loss and softmax loss.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def pair_weights(self, labels, ranks):
    """Returns the weight adjustment `Tensor` for example pairs.

    Args:
      labels: A dense `Tensor` of labels with shape [batch_size, list_size].
      ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
        sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    raise NotImplementedError('Calling an abstract method.')

  def individual_weights(self, labels, ranks):
    """Returns the weight `Tensor` for individual examples.

    Args:
      labels: A dense `Tensor` of labels with shape [batch_size, list_size].
      ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
        sorted by logits.

    Returns:
      A `Tensor` that can weight individual examples.
    """
    del ranks
    return labels


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

  def pair_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='dcg_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      valid_pair, labels = _get_valid_pairs_and_clean_labels(labels)
      gain = self._gain_fn(labels)
      if self._normalized:
        gain *= utils.inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      pair_gain = _apply_pairwise_op(tf.subtract, gain)
      pair_gain *= tf.cast(valid_pair, dtype=tf.float32)

      list_size = tf.shape(input=labels)[1]
      topn = self._topn or list_size

      def _discount_for_relative_rank_diff():
        """Rank-based discount in the LambdaLoss paper."""
        # The LambdaLoss is not well defined when topn is active and topn <
        # list_size. We cap the rank of examples to topn + 1 so that the rank
        # differene is capped to topn. This is just a convenient upperbound
        # when topn is active. We need to revisit this.
        capped_rank = tf.compat.v1.where(
            tf.greater(ranks, topn),
            tf.ones_like(ranks) * (topn + 1), ranks)
        rank_diff = tf.cast(
            tf.abs(_apply_pairwise_op(tf.subtract, capped_rank)),
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
            tf.greater(ranks, topn),
            tf.zeros_like(tf.cast(ranks, dtype=tf.float32)),
            self._rank_discount_fn(tf.cast(ranks, dtype=tf.float32)))
        pair_discount = tf.abs(_apply_pairwise_op(tf.subtract, rank_discount))
        return pair_discount

      u = _discount_for_relative_rank_diff()
      v = _discount_for_absolute_rank()
      pair_discount = (1. -
                       self._smooth_fraction) * u + self._smooth_fraction * v
      pair_weight = tf.abs(pair_gain) * pair_discount
      if self._topn is None:
        return pair_weight
      pair_mask = _apply_pairwise_op(tf.logical_or,
                                     tf.less_equal(ranks, self._topn))
      return pair_weight * tf.cast(pair_mask, dtype=tf.float32)

  def individual_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='dcg_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      labels = tf.convert_to_tensor(value=labels)
      labels = tf.compat.v1.where(
          utils.is_label_valid(labels), labels, tf.zeros_like(labels))
      gain = self._gain_fn(labels)
      if self._normalized:
        gain *= utils.inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      rank_discount = self._rank_discount_fn(tf.cast(ranks, dtype=tf.float32))
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

  def pair_weights(self, labels, ranks):
    """See `_LambdaWeight`.

    The current implementation here is that for any pairs of documents i and j,
    we set the weight to be 1 if
      - i and j have different labels.
      - i <= topn and j > topn or i > topn and j <= topn.
    This is exactly the same as the original LambdaRank method. The weight is
    the gain of swapping a pair of documents.

    Args:
      labels: A dense `Tensor` of labels with shape [batch_size, list_size].
      ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
        sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    with tf.compat.v1.name_scope(name='precision_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      valid_pair, labels = _get_valid_pairs_and_clean_labels(labels)
      binary_labels = tf.cast(self._positive_fn(labels), dtype=tf.float32)
      label_diff = tf.abs(_apply_pairwise_op(tf.subtract, binary_labels))
      label_diff *= tf.cast(valid_pair, dtype=tf.float32)
      # i <= topn and j > topn or i > topn and j <= topn, i.e., xor(i <= topn, j
      # <= topn).
      rank_mask = _apply_pairwise_op(tf.math.logical_xor,
                                     tf.less_equal(ranks, self._topn))
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

  def pair_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    pass

  def individual_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='p_list_mle_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      labels = tf.convert_to_tensor(value=labels)
      rank_discount = self._rank_discount_fn(tf.cast(ranks, dtype=tf.float32))
      return tf.ones_like(labels) * rank_discount


def _compute_ranks(logits, is_valid):
  """Computes ranks by sorting valid logits.

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    is_valid: A `Tensor` of the same shape as `logits` representing validity of
      each entry.

  Returns:
    The `ranks` Tensor.
  """
  _check_tensor_shapes([logits, is_valid])
  # Only sort entries with is_valid = True.
  scores = tf.compat.v1.where(
      is_valid, logits, -1e-6 * tf.ones_like(logits) +
      tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
  return utils.sorted_ranks(scores)


def _pairwise_comparison(labels, logits):
  r"""Returns pairwise comparison `Tensor`s.

  Given a list of n items, the labels of graded relevance l_i and the logits
  s_i, we form n^2 pairs. For each pair, we have the following:

                        /
                        | 1   if l_i > l_j for valid l_i and l_j.
  * `pairwise_labels` = |
                        | 0   otherwise
                        \
  * `pairwise_logits` = s_i - s_j

  Args:
    labels: A `Tensor` with shape [batch_size, list_size].
    logits: A `Tensor` with shape [batch_size, list_size].

  Returns:
    A tuple of (pairwise_labels, pairwise_logits) with each having the shape
    [batch_size, list_size, list_size].
  """
  # Compute the difference for all pairs in a list. The output is a Tensor with
  # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
  # the information for pair (i, j).
  pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
  pairwise_logits = _apply_pairwise_op(tf.subtract, logits)
  # Only keep the case when l_i > l_j.
  pairwise_labels = tf.cast(
      tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
  is_valid = utils.is_label_valid(labels)
  valid_pair = _apply_pairwise_op(tf.logical_and, is_valid)
  pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
  return pairwise_labels, pairwise_logits


class _RankingLoss(object):
  """Interface for ranking loss."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The loss name."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def compute_unreduced_loss(self, labels, logits):
    """Computes the unreduced loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.

    Returns:
      A tuple(losses, loss_weights) that have the same shape.
    """
    raise NotImplementedError('Calling an abstract method.')

  def normalize_weights(self, labels, weights):
    """Normalizes weights needed for tf.estimator (not tf.keras).

    This is needed for `tf.estimator` given that the reduction may be
    `SUM_OVER_NONZERO_WEIGHTS`. This function is not needed after we migrate
    from the deprecated reduction to `SUM` or `SUM_OVER_BATCH_SIZE`.

    Args:
      labels: A `Tensor` of shape [batch_size, list_size] representing graded
        relevance.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      The normalized weights.
    """
    del labels
    return 1.0 if weights is None else weights

  def compute(self, labels, logits, weights, reduction):
    """Computes the reduced loss for tf.estimator (not tf.keras).

    Note that this function is not compatible with keras.

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

    Returns:
      Reduced loss for training and eval.
    """
    losses, loss_weights = self.compute_unreduced_loss(labels, logits)
    weights = tf.multiply(self.normalize_weights(labels, weights), loss_weights)
    return tf.compat.v1.losses.compute_weighted_loss(
        losses, weights, reduction=reduction)

  def eval_metric(self, labels, logits, weights):
    """Computes the eval metric for the loss in tf.estimator (not tf.keras).

    Note that this function is not compatible with keras.

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
    losses, loss_weights = self.compute_unreduced_loss(labels, logits)
    weights = tf.multiply(self.normalize_weights(labels, weights), loss_weights)
    return tf.compat.v1.metrics.mean(losses, weights)


class _PairwiseLoss(_RankingLoss):
  """Interface for pairwise ranking loss."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, name, lambda_weight=None, params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      params: A dict for params used in loss computation.
    """
    self._name = name
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

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    is_valid = utils.is_label_valid(labels)
    ranks = _compute_ranks(logits, is_valid)
    pairwise_labels, pairwise_logits = _pairwise_comparison(labels, logits)
    pairwise_weights = pairwise_labels
    if self._lambda_weight is not None:
      pairwise_weights *= self._lambda_weight.pair_weights(labels, ranks)
      # For LambdaLoss with relative rank difference, the scale of loss becomes
      # much smaller when applying LambdaWeight. This affects the training can
      # make the optimal learning rate become much larger. We use a heuristic to
      # scale it up to the same magnitude as standard pairwise loss.
      pairwise_weights *= tf.cast(tf.shape(input=labels)[1], dtype=tf.float32)

    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return self._pairwise_loss(pairwise_logits), pairwise_weights

  def normalize_weights(self, labels, weights):
    """See _RankingLoss."""
    # The `weights` is item-wise and is applied non-symmetrically to update
    # pairwise_weights as
    #   pairwise_weights(i, j) = w_i * pairwise_weights(i, j).
    # This effectively applies to all pairs with l_i > l_j. Note that it is
    # actually symmetric when `weights` are constant per list, i.e., listwise
    # weights.
    if weights is None:
      weights = 1.
    weights = tf.compat.v1.where(
        utils.is_label_valid(labels),
        tf.ones_like(labels) * weights, tf.zeros_like(labels))
    return tf.expand_dims(weights, axis=2)


class PairwiseLogisticLoss(_PairwiseLoss):
  """Implements pairwise logistic loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    # The following is the same as log(1 + exp(-pairwise_logits)).
    return tf.nn.relu(-pairwise_logits) + tf.math.log1p(
        tf.exp(-tf.abs(pairwise_logits)))


class PairwiseHingeLoss(_PairwiseLoss):
  """Implements pairwise hinge loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    return tf.nn.relu(1 - pairwise_logits)


class PairwiseSoftZeroOneLoss(_PairwiseLoss):
  """Implements pairwise hinge loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    return tf.compat.v1.where(
        tf.greater(pairwise_logits, 0), 1. - tf.sigmoid(pairwise_logits),
        tf.sigmoid(-pairwise_logits))


class _ListwiseLoss(_RankingLoss):
  """Interface for listwise loss."""

  def __init__(self, name, lambda_weight=None, params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      params: A dict for params used in loss computation.
    """
    self._name = name
    self._lambda_weight = lambda_weight
    self._params = params or {}

  @property
  def name(self):
    """The loss name."""
    return self._name

  def normalize_weights(self, labels, weights):
    """See `_RankingLoss`."""
    if weights is None:
      return 1.0
    else:
      weights = tf.convert_to_tensor(value=weights)
      labels = tf.convert_to_tensor(value=labels)
      return tf.compat.v1.math.divide_no_nan(
          tf.reduce_sum(input_tensor=(weights * labels), axis=1, keepdims=True),
          tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True))


class SoftmaxLoss(_ListwiseLoss):
  """Implements softmax loss."""

  def precompute(self, labels, logits, weights):
    """Precomputes Tensors for softmax cross entropy inputs."""
    is_valid = utils.is_label_valid(labels)
    ranks = _compute_ranks(logits, is_valid)
    # Reset the invalid labels to 0 and reset the invalid logits to a logit with
    # ~= 0 contribution in softmax.
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(is_valid, logits,
                                tf.math.log(_EPSILON) * tf.ones_like(logits))
    if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                      DCGLambdaWeight):
      labels = self._lambda_weight.individual_weights(labels, ranks)
    if weights is not None:
      labels *= weights
    return labels, logits

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    # Padding for rows with label_sum = 0.
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    padded_labels = tf.compat.v1.where(nonzero_mask, labels,
                                       _EPSILON * tf.ones_like(labels))
    padded_label_sum = tf.reduce_sum(
        input_tensor=padded_labels, axis=1, keepdims=True)
    labels_for_softmax = padded_labels / padded_label_sum
    logits_for_softmax = logits
    # Padded labels have 0 weights in label_sum.
    weights_for_softmax = tf.reshape(label_sum, [-1])
    losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels_for_softmax, logits_for_softmax)
    return losses, weights_for_softmax

  def compute(self, labels, logits, weights, reduction):
    """See `_RankingLoss`."""
    labels, logits = self.precompute(labels, logits, weights)
    losses, weights = self.compute_unreduced_loss(labels, logits)
    return tf.compat.v1.losses.compute_weighted_loss(
        losses, weights, reduction=reduction)

  def eval_metric(self, labels, logits, weights):
    """See `_RankingLoss`."""
    labels, logits = self.precompute(labels, logits, weights)
    losses, weights = self.compute_unreduced_loss(labels, logits)
    return tf.compat.v1.metrics.mean(losses, weights)


class _PointwiseLoss(_RankingLoss):
  """Interface for pointwise loss."""

  def __init__(self, name, params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      params: A dict for params used in loss computation.
    """
    self._name = name
    self._params = params or {}

  @property
  def name(self):
    """The loss name."""
    return self._name

  def normalize_weights(self, labels, weights):
    """See _RankingLoss."""
    if weights is None:
      weights = 1.
    return tf.compat.v1.where(
        utils.is_label_valid(labels),
        tf.ones_like(labels) * weights, tf.zeros_like(labels))


class SigmoidCrossEntropyLoss(_PointwiseLoss):
  """Implements sigmoid cross entropy loss."""

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    labels = tf.compat.v1.where(
        utils.is_label_valid(labels), labels, tf.zeros_like(labels))
    losses = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return losses, 1.


class MeanSquaredLoss(_PointwiseLoss):
  """Implements the means squared error loss."""

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    is_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(is_valid, logits, tf.zeros_like(logits))
    losses = tf.compat.v1.squared_difference(labels, logits)
    return losses, 1.


class ListMLELoss(_ListwiseLoss):
  """Implements ListMLE loss."""

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    is_valid = utils.is_label_valid(labels)
    # Reset the invalid labels to 0 and reset the invalid logits to a logit with
    # ~= 0 contribution.
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(is_valid, logits,
                                tf.math.log(_EPSILON) * tf.ones_like(logits))
    scores = tf.compat.v1.where(
        is_valid, labels,
        tf.reduce_min(labels, axis=1, keepdims=True) -
        1e-6 * tf.ones_like(labels))
    # Use a fixed ops-level seed and the randomness is controlled by the
    # graph-level seed.
    sorted_labels, sorted_logits = utils.sort_by_scores(
        scores, [labels, logits], shuffle_ties=True, seed=37)

    raw_max = tf.reduce_max(input_tensor=sorted_logits, axis=1, keepdims=True)
    sorted_logits = sorted_logits - raw_max
    sums = tf.cumsum(tf.exp(sorted_logits), axis=1, reverse=True)
    sums = tf.math.log(sums) - sorted_logits

    if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                      ListMLELambdaWeight):
      batch_size, list_size = tf.unstack(tf.shape(sorted_labels))
      sums *= self._lambda_weight.individual_weights(
          sorted_labels,
          tf.tile(tf.expand_dims(tf.range(list_size) + 1, 0), [batch_size, 1]))

    negative_log_likelihood = tf.reduce_sum(
        input_tensor=sums, axis=1, keepdims=True)
    return negative_log_likelihood, 1.


class ApproxNDCGLoss(_ListwiseLoss):
  """Implements ApproxNDCG loss."""

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    alpha = self._params.get('alpha', 10.0)
    is_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
        tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    gains = tf.pow(2., tf.cast(labels, dtype=tf.float32)) - 1.
    ranks = utils.approx_ranks(logits, alpha=alpha)
    discounts = 1. / tf.math.log1p(ranks)
    dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1, keepdims=True)
    cost = -dcg * utils.inverse_max_dcg(labels)
    return cost, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])


class ApproxMRRLoss(_ListwiseLoss):
  """Implements ApproxMRR loss."""

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    alpha = self._params.get('alpha', 10.0)
    is_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
        tf.math.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.math.reduce_sum(input_tensor=labels, axis=1, keepdims=True)

    nonzero_mask = tf.math.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))

    rr = 1. / utils.approx_ranks(logits, alpha=alpha)
    rr = tf.math.reduce_sum(input_tensor=rr * labels, axis=-1, keepdims=True)
    mrr = rr / tf.math.reduce_sum(input_tensor=labels, axis=-1, keepdims=True)
    return -mrr, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])
