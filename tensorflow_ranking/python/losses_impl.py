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

"""Implements the losses for TF-Ranking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math

import tensorflow as tf

from tensorflow_ranking.python import utils

# The smallest probability that is used to derive smallest logit for invalid or
# padding entries.
_EPSILON = 1e-10


def _safe_default_gain_fn(labels):
  """Calculates safe gain functions for NDCG.

  In applications such as distillation, the labels could have extreme values
  that might result in numerical error when using the original gain function.
  This should only be applied to NDCG related losses, but not DCG ones. It
  should be applied on both the numerator and the denominator of NDCG.

  Args:
    labels: A `Tensor` with shape [batch_size, list_size], representing graded
      relevance.
  Returns:
    A `tensor` of safe gain function values of shape [batch_size, list_size].
  """
  max_labels = tf.reduce_max(labels, axis=-1, keepdims=True)
  gains = tf.pow(2., labels - max_labels) - tf.pow(2., -max_labels)
  return gains


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


def approx_ranks(logits):
  r"""Computes approximate ranks given a list of logits.

  Given a list of logits, the rank of an item in the list is one plus the total
  number of items with a larger logit. In other words,

    rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

  where "I" is the indicator function. The indicator function can be
  approximated by a generalized sigmoid:

    I_{s_j < s_i} \approx 1/(1 + exp(-(s_j - s_i)/temperature)).

  This function approximates the rank of an item using this sigmoid
  approximation to the indicator function. This technique is at the core
  of "A general approximation framework for direct optimization of
  information retrieval measures" by Qin et al.

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.

  Returns:
    A `Tensor` of ranks with the same shape as logits.
  """
  list_size = tf.shape(input=logits)[1]
  x = tf.tile(tf.expand_dims(logits, 2), [1, 1, list_size])
  y = tf.tile(tf.expand_dims(logits, 1), [1, list_size, 1])
  pairs = tf.sigmoid(y - x)
  return tf.reduce_sum(input_tensor=pairs, axis=-1) + .5


def inverse_max_dcg(labels,
                    gain_fn=lambda labels: tf.pow(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / tf.math.log1p(rank),
                    topn=None):
  """Computes the inverse of max DCG.

  Args:
    labels: A `Tensor` with shape [batch_size, list_size]. Each value is the
      graded relevance of the corresponding item.
    gain_fn: A gain function. By default this is set to: 2^label - 1.
    rank_discount_fn: A discount function. By default this is set to:
      1/log(1+rank).
    topn: An integer as the cutoff of examples in the sorted list.

  Returns:
    A `Tensor` with shape [batch_size, 1].
  """
  ideal_sorted_labels, = utils.sort_by_scores(labels, [labels], topn=topn)
  rank = tf.range(tf.shape(input=ideal_sorted_labels)[1]) + 1
  discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(
      tf.cast(rank, dtype=tf.float32))
  discounted_gain = tf.reduce_sum(
      input_tensor=discounted_gain, axis=1, keepdims=True)
  return tf.compat.v1.where(
      tf.greater(discounted_gain, 0.), 1. / discounted_gain,
      tf.zeros_like(discounted_gain))


def ndcg(labels, ranks=None, perm_mat=None):
  """Computes NDCG from labels and ranks.

  Args:
    labels: A `Tensor` with shape [batch_size, list_size], representing graded
      relevance.
    ranks: A `Tensor` of the same shape as labels, or [1, list_size], or None.
      If ranks=None, we assume the labels are sorted in their rank.
    perm_mat: A `Tensor` with shape [batch_size, list_size, list_size] or None.
      Permutation matrices with rows correpond to the ranks and columns
      correspond to the indices. An argmax over each row gives the index of the
      element at the corresponding rank.

  Returns:
    A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
  """
  if ranks is not None and perm_mat is not None:
    raise ValueError('Cannot use both ranks and perm_mat simultaneously.')

  if ranks is None:
    list_size = tf.shape(labels)[1]
    ranks = tf.range(list_size) + 1
  discounts = 1. / tf.math.log1p(tf.cast(ranks, dtype=tf.float32))
  gains = _safe_default_gain_fn(tf.cast(labels, dtype=tf.float32))
  if perm_mat is not None:
    gains = tf.reduce_sum(
        input_tensor=perm_mat * tf.expand_dims(gains, 1), axis=-1)
  dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1, keepdims=True)
  normalized_dcg = dcg * inverse_max_dcg(labels, gain_fn=_safe_default_gain_fn)

  return normalized_dcg


class _LambdaWeight(object, metaclass=abc.ABCMeta):
  """Interface for ranking metric optimization.

  This class wraps weights used in the LambdaLoss framework for ranking metric
  optimization (https://ai.google/research/pubs/pub47258). Such an interface is
  to be instantiated by concrete lambda weight models. The instance is used
  together with standard loss such as logistic loss and softmax loss.
  """
  # TODO: Define a public version of `_LambdaWeight` for typing
  # annotations.

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


class LabelDiffLambdaWeight(_LambdaWeight):
  """A simple LambdaWeight to compute the pair label difference."""

  def pair_weights(self, labels, ranks):
    """Returns the absolute label difference for each pair."""
    del ranks  # Unused.
    return tf.abs(_apply_pairwise_op(tf.subtract, labels))


class AbstractDCGLambdaWeight(_LambdaWeight):
  """Abstract LambdaWeight for Discounted Cumulative Gain (DCG) metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False):
    """Initializer.

    Ranks are 1-based, not 0-based.

    Args:
      topn: (int) The topn for the DCG metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
      normalized: (bool) If True, normalize weight by the max DCG.
    """
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._normalized = normalized

  @abc.abstractmethod
  def _pair_rank_discount(self, ranks, topn):
    """Computes the rank-based discount for a pair.

    Args:
      ranks: A 2D `Tensor` for the 1-based ranks.
      topn: A scalar `Tensor` for the topn cutoff.

    Returns:
     A pairwise weights `Tensor` based on the `rank_discount_fn`.
    """
    raise NotImplementedError('Calling an abstract method.')

  def pair_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='dcg_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      valid_pair, labels = _get_valid_pairs_and_clean_labels(labels)
      gain = self._gain_fn(labels)
      if self._normalized:
        gain *= inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      pair_gain = _apply_pairwise_op(tf.subtract, gain)
      pair_gain *= tf.cast(valid_pair, dtype=tf.float32)

      list_size = tf.shape(input=labels)[1]
      topn = self._topn or list_size
      pair_weight = tf.abs(pair_gain) * self._pair_rank_discount(ranks, topn)

      # For LambdaLoss with relative rank difference, the scale of loss becomes
      # much smaller when applying LambdaWeight. This affects the training can
      # make the optimal learning rate become much larger. We use a heuristic to
      # scale it up to the same magnitude as standard pairwise loss.
      pair_weight *= tf.cast(tf.shape(input=labels)[1], dtype=tf.float32)
      return pair_weight

  def individual_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    with tf.compat.v1.name_scope(name='dcg_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      labels = tf.convert_to_tensor(value=labels)
      labels = tf.compat.v1.where(
          utils.is_label_valid(labels), labels, tf.zeros_like(labels))
      gain = self._gain_fn(labels)
      if self._normalized:
        gain *= inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      rank_discount = self._rank_discount_fn(tf.cast(ranks, dtype=tf.float32))
      return gain * rank_discount


class DCGLambdaWeight(AbstractDCGLambdaWeight):
  """LambdaWeight for Discounted Cumulative Gain metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False,
               smooth_fraction=0.):
    """Initializer.

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
    super().__init__(topn, gain_fn, rank_discount_fn, normalized)
    if not 0. <= smooth_fraction <= 1.:
      raise ValueError('smooth_fraction %s should be in range [0, 1].' %
                       smooth_fraction)
    self._smooth_fraction = smooth_fraction

  def _pair_rank_discount(self, ranks, topn):
    """See `_LambdaWeight`."""

    def _discount_for_relative_rank_diff():
      """Rank-based discount in the LambdaLoss paper."""
      # The LambdaLoss is not well defined when topn is active and topn <
      # list_size. The following implementation is based on Equation 18 proposed
      # in https://research.google/pubs/pub47258/. Please refer to
      # `DCGLambdaWeightV2` for a better implemention to handle topn.
      pair_valid_rank = _apply_pairwise_op(tf.logical_or,
                                           tf.less_equal(ranks, topn))
      rank_diff = tf.cast(
          tf.abs(_apply_pairwise_op(tf.subtract, ranks)), dtype=tf.float32)
      pair_discount = tf.where(
          tf.logical_and(tf.greater(rank_diff, 0), pair_valid_rank),
          tf.abs(
              self._rank_discount_fn(rank_diff) -
              self._rank_discount_fn(rank_diff + 1)), tf.zeros_like(rank_diff))
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
    pair_discount = (1. - self._smooth_fraction) * u + self._smooth_fraction * v
    pair_mask = _apply_pairwise_op(tf.logical_or, tf.less_equal(ranks, topn))
    return pair_discount * tf.cast(pair_mask, dtype=tf.float32)


class DCGLambdaWeightV2(AbstractDCGLambdaWeight):
  """The V2 version of LambdaWeight for DCG metric.

  V2: Everything is the same as LambdaLoss when topn=None. When topn is
  activated, for any pair i, j where max(i, j) > topn, we multiply the inverse
  of 1-1/log(1+max(i,j)) for example.
  """

  def _pair_rank_discount(self, ranks, topn):
    """Implements the rank discount for pairs in topn metrics."""
    rank_diff = tf.cast(
        tf.abs(_apply_pairwise_op(tf.subtract, ranks)), dtype=tf.float32)
    max_rank = tf.cast(_apply_pairwise_op(tf.math.maximum, ranks), tf.float32)
    multiplier = tf.where(
        tf.greater(max_rank, tf.cast(topn, tf.float32)),
        1. / (1. - self._rank_discount_fn(max_rank)), 1.)
    pair_discount = tf.where(
        tf.greater(rank_diff, 0.),
        tf.abs(
            self._rank_discount_fn(rank_diff) -
            self._rank_discount_fn(rank_diff + 1)) * multiplier,
        tf.zeros_like(rank_diff))
    return pair_discount


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


def _pairwise_comparison(labels, logits, mask, pairwise_logits_op=tf.subtract):
  r"""Returns pairwise comparison `Tensor`s.

  Given a list of n items, the labels of graded relevance l_i and the logits
  s_i, we form n^2 pairs. For each pair, we have the following:

                        /
                        | 1   if l_i > l_j for valid l_i and l_j.
  * `pairwise_labels` = |
                        | 0   otherwise
                        \
  * `pairwise_logits` = pairwise_logits_op(s_i, s_j)

  Args:
    labels: A `Tensor` with shape [batch_size, list_size].
    logits: A `Tensor` with shape [batch_size, list_size].
    mask: A `Tensor` with shape [batch_size, list_size] indicating which entries
      are valid for computing the pairwise comparisons.
    pairwise_logits_op: A pairwise function which operates on 2 tensors.

  Returns:
    A tuple of (pairwise_labels, pairwise_logits) with each having the shape
    [batch_size, list_size, list_size].
  """
  # Compute the difference for all pairs in a list. The output is a Tensor with
  # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
  # the information for pair (i, j).
  pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
  pairwise_logits = _apply_pairwise_op(pairwise_logits_op, logits)
  # Only keep the case when l_i > l_j.
  pairwise_labels = tf.cast(
      tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
  valid_pair = _apply_pairwise_op(tf.logical_and, mask)
  pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
  return pairwise_labels, pairwise_logits


class GumbelSampler(object):
  """Random sampler for sampling gumbel distributed logits."""

  def __init__(self,
               name=None,
               sample_size=8,
               temperature=1.0,
               seed=None,
               ragged=False):
    """Constructor."""
    self._name = name
    self._sample_size = sample_size
    self._temperature = temperature
    self._seed = seed
    self._ragged = ragged

  def sample(self, labels, logits, weights=None):
    """Samples scores from Concrete(logits).

    If the sampler was constructed with `ragged=True` this method expects
    `labels`, `logits` and item-wise `weights` to be a `RaggedTensor`.

    Args:
      labels: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size]
        same as `logits`, representing graded relevance. Or in the diversity
        tasks, a `Tensor` (or `RaggedTensor`) with shape [batch_size, list_size,
        subtopic_size]. Each value represents relevance to a subtopic, 1 for
        relevent subtopic, 0 for irrelevant, and -1 for paddings. When the
        actual subtopic number of a query is smaller than the `subtopic_size`,
        `labels` will be padded to `subtopic_size` with -1.
      logits: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size].
        Each value is the ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` or `RaggedTensor` with shape [batch_size,
        list_size] for item-wise weights. If None, the weight of a list in the
        mini-batch is set to the sum of the labels of the items in that list.

    Returns:
      A tuple of expanded labels, logits, and weights where the first dimension
      is now batch_size * sample_size. Logit Tensors are sampled from
      Concrete(logits) while labels and weights are simply tiled so the
      resulting
      Tensor has the updated dimensions.
    """
    with tf.compat.v1.name_scope(self._name, 'gumbel_softmax_sample',
                                 (labels, logits, weights)):
      # Convert ragged tensors to dense and construct a mask.
      if self._ragged:
        is_weights_ragged = isinstance(weights, tf.RaggedTensor)
        labels, logits, weights, mask = utils.ragged_to_dense(
            labels, logits, weights)

      batch_size = tf.shape(input=labels)[0]
      list_size = tf.shape(input=labels)[1]

      # Expand labels.
      expanded_labels = tf.expand_dims(labels, 1)
      expanded_labels = tf.repeat(expanded_labels, [self._sample_size], axis=1)
      expanded_labels = utils.reshape_first_ndims(
          expanded_labels, 2, [batch_size * self._sample_size])

      # Sample logits from Concrete(logits).
      sampled_logits = tf.expand_dims(logits, 1)
      sampled_logits = tf.tile(sampled_logits, [1, self._sample_size, 1])
      sampled_logits += _sample_gumbel(
          [batch_size, self._sample_size, list_size], seed=self._seed)
      sampled_logits = tf.reshape(sampled_logits,
                                  [batch_size * self._sample_size, list_size])

      is_label_valid = utils.is_label_valid(expanded_labels)
      if is_label_valid.shape.rank > 2:
        is_label_valid = tf.reduce_any(is_label_valid, axis=-1)
      sampled_logits = tf.compat.v1.where(
          is_label_valid, sampled_logits / self._temperature,
          tf.math.log(1e-20) * tf.ones_like(sampled_logits))
      sampled_logits = tf.math.log(tf.nn.softmax(sampled_logits) + 1e-20)

      expanded_weights = weights
      if expanded_weights is not None:
        true_fn = lambda: tf.expand_dims(tf.expand_dims(expanded_weights, 1), 1)
        false_fn = lambda: tf.expand_dims(expanded_weights, 1)
        expanded_weights = tf.cond(
            pred=tf.math.equal(tf.rank(expanded_weights), 1),
            true_fn=true_fn,
            false_fn=false_fn)
        expanded_weights = tf.tile(expanded_weights, [1, self._sample_size, 1])
        expanded_weights = tf.reshape(expanded_weights,
                                      [batch_size * self._sample_size, -1])

      # Convert dense tensors back to ragged.
      if self._ragged:
        # Construct expanded mask for the number of samples.
        expanded_mask = tf.expand_dims(mask, 1)
        expanded_mask = tf.repeat(expanded_mask, [self._sample_size], axis=1)
        expanded_mask = tf.reshape(expanded_mask,
                                   [batch_size * self._sample_size, list_size])
        # Convert labels and sampled logits to ragged tensors.
        expanded_labels = tf.ragged.boolean_mask(expanded_labels, expanded_mask)
        sampled_logits = tf.ragged.boolean_mask(sampled_logits, expanded_mask)
        # If ragged weights were provided, convert dense weights back to ragged.
        if is_weights_ragged:
          expanded_weights = tf.ragged.boolean_mask(expanded_weights,
                                                    expanded_mask)

      return expanded_labels, sampled_logits, expanded_weights


def _sample_gumbel(shape, eps=1e-20, seed=None):
  u = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32, seed=seed)
  return -tf.math.log(-tf.math.log(u + eps) + eps)


class _RankingLoss(object, metaclass=abc.ABCMeta):
  """Interface for ranking loss."""

  def __init__(self, name, lambda_weight=None, temperature=1.0, ragged=False):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      temperature: A float number to modify the logits=logits/temperature.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    self._name = name
    self._lambda_weight = lambda_weight
    self._temperature = temperature
    self._ragged = ragged

  @property
  def name(self):
    """The loss name."""
    return self._name

  def _prepare_and_validate_params(self, labels, logits, weights, mask):
    """Prepares and validate input parameters.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the loss.

    Returns:
      A tuple (labels, logits, weights, mask) of `tf.Tensor` objects that are
      ready to be used in the loss.
    """
    if self._ragged:
      labels, logits, weights, mask = utils.ragged_to_dense(
          labels, logits, weights)

    if mask is None:
      mask = utils.is_label_valid(labels)

    if weights is None:
      weights = 1.0

    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    weights = tf.convert_to_tensor(weights)
    mask = tf.convert_to_tensor(mask)

    return labels, logits, weights, mask

  def compute_unreduced_loss(self, labels, logits, mask=None):
    """Computes the unreduced loss.

    Args:
      labels: A `Tensor` or `RaggedTensor` of the same shape as `logits`
        representing graded relevance.
      logits: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size].
        Each value is the ranking score of the corresponding item.
      mask: An optional `Tensor` of the same shape as logits indicating which
        entries are valid for computing the loss. Will be ignored if the loss
        was constructed with ragged=True.

    Returns:
      A tuple(losses, loss_weights) that have the same shape.
    """
    labels, logits, _, mask = self._prepare_and_validate_params(
        labels, logits, None, mask)
    return self._compute_unreduced_loss_impl(labels, logits, mask)

  @abc.abstractmethod
  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """Implementation for the unreduced loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      mask: An optional `Tensor` of the same shape as logits indicating which
        entries are valid for computing the loss.

    Returns:
      A tuple(losses, loss_weights) that have the same shape.
    """
    raise NotImplementedError('Calling an abstract method.')

  def normalize_weights(self, labels, weights):
    """Normalizes weights.

    This is needed for `tf.estimator` given that the reduction may be
    `SUM_OVER_NONZERO_WEIGHTS`.

    This method is also needed to compute normalized weights when calling
    `compute_unreduced_loss`, which is done in the tf.keras losses.

    Args:
      labels: A `Tensor` of shape [batch_size, list_size] representing graded
        relevance.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      The normalized weights.
    """
    if self._ragged:
      labels, _, weights, _ = utils.ragged_to_dense(labels, None, weights)
    return self._normalize_weights_impl(labels, weights)

  def _normalize_weights_impl(self, labels, weights):
    """See `normalize_weights`."""
    del labels
    return 1.0 if weights is None else weights

  def get_logits(self, logits):
    """Computes logits rescaled by temperature.

    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.

    Returns:
      Tensor of rescaled logits.
    """
    if not tf.is_tensor(logits):
      logits = tf.convert_to_tensor(value=logits)
    return logits / self._temperature

  def compute(self, labels, logits, weights, reduction, mask=None):
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
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the loss.

    Returns:
      Reduced loss for training and eval.
    """
    logits = self.get_logits(logits)
    losses, loss_weights = self._compute_unreduced_loss_impl(
        labels, logits, mask)
    weights = tf.multiply(
        self._normalize_weights_impl(labels, weights), loss_weights)
    return tf.compat.v1.losses.compute_weighted_loss(
        losses, weights, reduction=reduction)

  @abc.abstractmethod
  def compute_per_list(self, labels, logits, weights, mask=None):
    """Computes the per-list loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the loss.

    Returns:
      A pair of `Tensor` objects of shape [batch_size] containing per-list
      losses and weights.
    """
    raise NotImplementedError('Calling an abstract method.')

  def eval_metric(self, labels, logits, weights, mask=None):
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
      mask: A `Tensor` of the same shape as logits indicating which entries are
        valid for computing the metric.

    Returns:
      A metric op.
    """
    losses, loss_weights = self._compute_unreduced_loss_impl(
        labels, logits, mask)
    weights = tf.multiply(
        self._normalize_weights_impl(labels, weights), loss_weights)
    return tf.compat.v1.metrics.mean(losses, weights)


class _PairwiseLoss(_RankingLoss, metaclass=abc.ABCMeta):
  """Interface for pairwise ranking loss."""

  @abc.abstractmethod
  def _pairwise_loss(self, pairwise_logits):
    """The loss of pairwise logits with l_i > l_j."""
    raise NotImplementedError('Calling an abstract method.')

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    ranks = _compute_ranks(logits, mask)
    pairwise_labels, pairwise_logits = _pairwise_comparison(
        labels, logits, mask)
    pairwise_weights = pairwise_labels
    if self._lambda_weight is not None:
      pairwise_weights *= self._lambda_weight.pair_weights(labels, ranks)

    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return self._pairwise_loss(pairwise_logits), pairwise_weights

  def compute_per_list(self, labels, logits, weights, mask=None):
    """See `_RankingLoss`."""
    # Prepare input params.
    labels, logits, weights, mask = self._prepare_and_validate_params(
        labels, logits, weights, mask)

    # Pairwise losses and weights will be of shape
    # [batch_size, list_size, list_size].
    losses, loss_weights = self._compute_unreduced_loss_impl(
        labels, logits, mask)
    weights = tf.multiply(
        self._normalize_weights_impl(labels, weights), loss_weights)

    # Compute the weighted per-pair loss.
    weighted_per_pair_loss = tf.math.multiply(losses, weights)

    # Sum the inner dimensions to obtain per-list weights. For pairwise losses
    # this typically indicates the (weighted) number of pairwise preferences per
    # list.
    per_list_weights = tf.reduce_sum(weights, axis=[1, 2])

    # This computes the per-list losses by summing all weighted pairwise losses.
    per_list_losses = tf.reduce_sum(weighted_per_pair_loss, axis=[1, 2])

    # Normalize the per-list losses so that lists with different numbers of
    # pairs have comparable losses. The different numbers of pairs is reflected
    # in the per-list weights.
    per_list_losses = tf.math.divide_no_nan(per_list_losses, per_list_weights)

    return per_list_losses, per_list_weights

  def _normalize_weights_impl(self, labels, weights):
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


class PairwiseMSELoss(_PairwiseLoss):
  """Implements pairwise MSE loss.

  This loss computes over all pairs, including those with the same labels, but
  excluding self pairs in the diagonal of the pairwise matrix.
  """

  def _pairwise_loss(self, pairwise_logits):
    # Unused because of overridding `_compute_unreduced_loss_impl`.
    pass

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)

    # Compute loss.
    pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
    pairwise_logit_diff = _apply_pairwise_op(tf.subtract, logits)
    pairwise_mse_loss = tf.math.square(pairwise_logit_diff -
                                       pairwise_label_diff)
    valid_pair = _apply_pairwise_op(tf.logical_and, mask)

    # Compute weights.
    pairwise_weights = tf.ones_like(pairwise_mse_loss)
    batch_size, list_size = tf.unstack(tf.shape(input=labels))
    # Excluding the self pairs.
    pairwise_weights -= tf.eye(
        list_size, batch_shape=[batch_size], dtype=pairwise_weights.dtype)
    # Including only valid pairs
    pairwise_weights *= tf.cast(valid_pair, tf.float32)
    if self._lambda_weight is not None:
      ranks = _compute_ranks(logits, mask)
      pairwise_weights *= self._lambda_weight.pair_weights(labels, ranks)
    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')

    return pairwise_mse_loss, pairwise_weights


class _ListwiseLoss(_RankingLoss):
  """Interface for listwise loss."""

  def _normalize_weights_impl(self, labels, weights):
    """See `_RankingLoss`."""
    if weights is None:
      return 1.0
    else:
      weights = tf.convert_to_tensor(value=weights)
      labels = tf.convert_to_tensor(value=labels)
      is_valid = utils.is_label_valid(labels)
      labels = tf.where(is_valid, labels, tf.zeros_like(labels))
      return tf.compat.v1.math.divide_no_nan(
          tf.reduce_sum(input_tensor=(weights * labels), axis=1, keepdims=True),
          tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True))

  def compute_per_list(self, labels, logits, weights, mask=None):
    """See `_RankingLoss`."""
    # Prepare input params.
    labels, logits, weights, mask = self._prepare_and_validate_params(
        labels, logits, weights, mask)

    # Listwise losses and weights will be of shape [batch_size, 1].
    losses, loss_weights = self._compute_unreduced_loss_impl(
        labels, logits, mask)
    weights = tf.multiply(
        self._normalize_weights_impl(labels, weights), loss_weights)

    # This removes the inner dimension of size 1 to make the output shape
    # [batch_size].
    per_list_losses = tf.squeeze(losses, axis=1)
    per_list_weights = tf.squeeze(weights, axis=1)
    return per_list_losses, per_list_weights


class CircleLoss(_ListwiseLoss):
  """Implements circle loss.

  This is the Circle loss originally proposed by Sun et al.
  ["Circle Loss: A Unified Perspective of Pair Similarity Optimization"]. See
  https://arxiv.org/abs/2002.10857.

  For a model that outputs similarity scores `s` on data point with
  corresponding label y, the circle loss from Eq.(6) in the paper is
    L_circle = log(1 + sum_{i is p,j is n}
                   exp(gamma * (a_j * (s_j - d_n) - a_i * (s_i - d_p)))),
  defined for the binary label, p for data points with positive labels and n for
  data points with negative labels.
    a_i = relu(1 + margin - s_i)
    a_j = relu(s_j + margin)
    d_p = 1 - margin
    d_n = margin
  We can extend to non-binary labels with an indiactor function,
    L_circle = log(1 + sum_{i, j} I_{y_i > y_j}
                   exp(gamma * (a_j * (s_j - d_n) - a_i * (s_i - d_p)))),
  Note the loss takes only the similarity scores. We will clip any score value
  beyond 0 and 1 to confine the scores in [0, 1], please be aware of that.
  """

  def __init__(self,
               name,
               lambda_weight=None,
               gamma=64,
               margin=0.25,
               ragged=False):
    """Initializer.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      gamma: A float parameter used in circle loss.
      margin: A float parameter defining the margin in circle loss.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    super().__init__(
        name, lambda_weight=lambda_weight, temperature=1.0, ragged=ragged)
    self._margin = margin
    self._gamma = gamma

  def get_logits(self, logits):
    """See `_RankingLoss`."""
    # Add a clip to confine scores in [0, 1].
    return tf.clip_by_value(tf.convert_to_tensor(value=logits), 0., 1.)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)

    def circle_loss_pairwise_op(score_i, score_j):
      alpha_i = tf.stop_gradient(
          tf.nn.relu(1 - score_i + self._margin), name='circle_loss_alpha_pos')
      alpha_j = tf.stop_gradient(
          tf.nn.relu(score_j + self._margin), name='circle_loss_alpha_neg')
      return alpha_i * (1 - score_i - self._margin) + alpha_j * (
          score_j - self._margin)

    pairwise_labels, pairwise_logits = _pairwise_comparison(
        labels, logits, mask, pairwise_logits_op=circle_loss_pairwise_op)
    pairwise_weights = tf.stop_gradient(
        pairwise_labels, name='weights_stop_gradient')
    # TODO: try lambda_weights for circle loss.
    # Pairwise losses and weights will be of shape
    # [batch_size, list_size, list_size].
    losses = tf.exp(self._gamma * pairwise_logits)

    # This computes the per-list losses and weights for circle loss.
    per_list_losses = tf.math.log1p(
        tf.reduce_sum(tf.math.multiply(losses, pairwise_weights), axis=[1, 2]))
    per_list_weights = tf.reduce_sum(
        pairwise_weights, axis=[1, 2]) / tf.reduce_sum(
            tf.cast(pairwise_weights > 0, tf.float32), axis=[1, 2])

    # Return per-list losses and weights with shape [batch_size, 1].
    return tf.expand_dims(per_list_losses,
                          1), tf.expand_dims(per_list_weights, 1)


class SoftmaxLoss(_ListwiseLoss):
  """Implements softmax loss."""

  def precompute(self, labels, logits, weights, mask=None):
    """Precomputes Tensors for softmax cross entropy inputs."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    ranks = _compute_ranks(logits, mask)
    # Reset the masked labels to 0 and reset the masked logits to a logit with
    # ~= 0 contribution in softmax.
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits,
                                tf.math.log(_EPSILON) * tf.ones_like(logits))
    if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                      DCGLambdaWeight):
      labels = self._lambda_weight.individual_weights(labels, ranks)
    if weights is not None:
      labels *= weights
    return labels, logits

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    # Padding for rows with label_sum = 0.
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    padded_labels = tf.compat.v1.where(nonzero_mask, labels,
                                       _EPSILON * tf.ones_like(labels))
    padded_labels = tf.compat.v1.where(mask, padded_labels,
                                       tf.zeros_like(padded_labels))
    padded_label_sum = tf.reduce_sum(
        input_tensor=padded_labels, axis=1, keepdims=True)
    labels_for_softmax = tf.math.divide_no_nan(padded_labels, padded_label_sum)
    logits_for_softmax = logits
    # Padded labels have 0 weights in label_sum.
    weights_for_softmax = tf.reshape(label_sum, [-1])
    losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels_for_softmax, logits_for_softmax)
    return losses, weights_for_softmax

  def compute(self, labels, logits, weights, reduction, mask=None):
    """See `_RankingLoss`."""
    labels, logits, weights, mask = self._prepare_and_validate_params(
        labels, logits, weights, mask)
    logits = self.get_logits(logits)
    labels, logits = self.precompute(labels, logits, weights, mask)
    losses, weights = self._compute_unreduced_loss_impl(labels, logits, mask)
    return tf.compat.v1.losses.compute_weighted_loss(
        losses, weights, reduction=reduction)

  def eval_metric(self, labels, logits, weights, mask=None):
    """See `_RankingLoss`."""
    labels, logits, weights, mask = self._prepare_and_validate_params(
        labels, logits, weights, mask)
    logits = self.get_logits(logits)
    labels, logits = self.precompute(labels, logits, weights, mask)
    losses, weights = self._compute_unreduced_loss_impl(labels, logits, mask)
    return tf.compat.v1.metrics.mean(losses, weights)

  def compute_per_list(self, labels, logits, weights, mask=None):
    """See `_RankingLoss`."""
    # Prepare input params.
    labels, logits, weights, mask = self._prepare_and_validate_params(
        labels, logits, weights, mask)

    # As opposed to the other listwise losses, SoftmaxLoss returns already
    # squeezed losses, which can be returned directly.
    logits = self.get_logits(logits)
    labels, logits = self.precompute(labels, logits, weights, mask)
    return self._compute_unreduced_loss_impl(labels, logits, mask)

  def compute_unreduced_loss(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    labels, logits, _, mask = self._prepare_and_validate_params(
        labels, logits, None, mask)
    logits = self.get_logits(logits)
    labels, logits = self.precompute(labels, logits, weights=None, mask=mask)
    return self._compute_unreduced_loss_impl(labels, logits, mask)


class PolyOneSoftmaxLoss(SoftmaxLoss):
  """Implements poly1 softmax loss."""

  def __init__(self,
               name,
               lambda_weight=None,
               epsilon=1.0,
               temperature=1.0,
               ragged=False):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      epsilon: A float number for contribution of the first polynomial.
      temperature: A float number to modify the logits=logits/temperature.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    super().__init__(
        name,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)
    self._epsilon = epsilon

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    # Padding for rows with label_sum = 0.
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    padded_labels = tf.compat.v1.where(nonzero_mask, labels,
                                       _EPSILON * tf.ones_like(labels))
    padded_labels = tf.compat.v1.where(mask, padded_labels,
                                       tf.zeros_like(padded_labels))
    padded_label_sum = tf.reduce_sum(
        input_tensor=padded_labels, axis=1, keepdims=True)
    labels_for_softmax = tf.math.divide_no_nan(padded_labels, padded_label_sum)
    logits_for_softmax = logits
    # Padded labels have 0 weights in label_sum.
    weights_for_softmax = tf.reshape(label_sum, [-1])
    pt = tf.reduce_sum(
        labels_for_softmax * tf.nn.softmax(logits_for_softmax), axis=-1)
    ce = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels_for_softmax, logits_for_softmax)
    losses = ce + self._epsilon * (1 - pt)
    return losses, weights_for_softmax


class UniqueSoftmaxLoss(_ListwiseLoss):
  """Implements unique rating softmax loss."""

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits,
                                tf.math.log(_EPSILON) * tf.ones_like(logits))
    pairwise_labels, _ = _pairwise_comparison(labels, logits, mask)
    # Used in denominator to compute unique softmax probability for each doc.
    denominator_logits = tf.expand_dims(logits, axis=1) * pairwise_labels
    denominator_logits = tf.concat(
        [denominator_logits, tf.expand_dims(logits, axis=2)], axis=2)
    denominator_mask = tf.concat(
        [pairwise_labels,
         tf.expand_dims(tf.ones_like(logits), axis=2)], axis=2)
    denominator_logits = tf.where(
        tf.greater(denominator_mask, 0.0), denominator_logits, -1e-3 +
        tf.reduce_min(denominator_logits) * tf.ones_like(denominator_logits))
    logits_max = tf.reduce_max(denominator_logits, axis=-1, keepdims=True)
    # Subtract the max so that exp(denominator_logits) is numerically valid.
    denominator_logits -= logits_max
    logits -= tf.squeeze(logits_max, axis=-1)
    # Set gains for loss weights.
    gains = tf.pow(2.0, labels) - 1
    # Compute the softmax loss for each doc.
    per_doc_softmax = -logits + tf.math.log(
        tf.reduce_sum(tf.exp(denominator_logits) * denominator_mask, axis=-1))
    losses = tf.reduce_sum(per_doc_softmax * gains, axis=1, keepdims=True)
    return losses, tf.ones_like(losses)


class _PointwiseLoss(_RankingLoss):
  """Interface for pointwise loss."""

  def _normalize_weights_impl(self, labels, weights):
    """See _RankingLoss."""
    if weights is None:
      weights = 1.
    return tf.compat.v1.where(
        utils.is_label_valid(labels),
        tf.ones_like(labels) * weights, tf.zeros_like(labels))

  def compute_per_list(self, labels, logits, weights, mask=None):
    """See `_RankingLoss`."""
    # Prepare input params.
    labels, logits, weights, mask = self._prepare_and_validate_params(
        labels, logits, weights, mask)

    # Pointwise losses and weights will be of shape [batch_size, list_size].
    losses, loss_weights = self._compute_unreduced_loss_impl(
        labels, logits, mask)
    weights = tf.multiply(
        self._normalize_weights_impl(labels, weights), loss_weights)

    # Compute the weighted per-item loss.
    weighted_per_item_loss = tf.math.multiply(losses, weights)

    # Sum the inner dimensions to obtain per-list weights. For pointwise losses
    # this typically indicates the (weighted) number of items per list.
    per_list_weights = tf.reduce_sum(weights, axis=1)

    # This computes the per-list losses by summing all weighted per-item losses.
    per_list_losses = tf.reduce_sum(weighted_per_item_loss, axis=1)

    # Normalize the per-list losses so that lists with different numbers of
    # items have comparable losses. The different numbers of items is reflected
    # in the per-list weights.
    per_list_losses = tf.math.divide_no_nan(per_list_losses, per_list_weights)
    return per_list_losses, per_list_weights


class ClickEMLoss(_PointwiseLoss):
  """Implements the click EM loss with examination and relevance.

  The implementation is based on the the paper by Wang et al: "Position bias
  estimation for unbiased learning to rank in personal search." It assumes that
  a click is generated by a factorized model P(examination) * P(relevance),
  which are latent variables determined by `exam_logits` and `rel_logits`
  respectively. An EM algorithm is used for estimation and this function
  implements the expectation step to estimate the P(latent | observed), i.e.,
  P(examination | click) and P(relevance | click).
  """

  def __init__(self,
               name,
               temperature=1.0,
               exam_loss_weight=1.0,
               rel_loss_weight=1.0,
               ragged=False):
    super().__init__(name, None, temperature, ragged)
    self._exam_loss_weight = exam_loss_weight
    self._rel_loss_weight = rel_loss_weight

  def _compute_latent_prob(self, clicks, exam_logits, rel_logits):
    """Computes the probability of latent variables in EM.

    The original compuation is as follows and can be unstable:
      exam_prob = sigmoid(exam_logits)
      rel_prob = sigmoid(rel_logits)
      exam_prob_posterior = exam_prob * (1 - rel_prob) / (1 - exam_prob *
        rel_prob)
      rel_prob_posterior = rel_prob * (1 - exam_prob) / (1 - exam_prob *
        rel_prob).

    To increase the numeric stability, we compute the posteriror logits first.
    Using the exam_logits_posterior as an example, we have:
      exam_logit_posterior = logit(exam_prob_posterior)
        = log(exam_prob_posterior / (1 - exam_prob_posterior))
    It can be reduced to exam_logits and rel_logits:
      exam_logit_posterior = exam_logits - log(1 + exp(rel_logits))
        = exam_logits - softplus(rel_logits)

    We can do similar reduction for rel_logit_posterior. Then we compute the
    posterior probablity by apply sigmoid on the logits.

    Args:
      clicks: A 2-D `Tensor` for clicks as observed data. A value >= 1.0 is
        treated as clicked.
      exam_logits: A 2-D `Tensor` to compute P(examination) and has the same
        shape as `clicks`.
      rel_logits: A 2-D `Tensor` to compute P(relevance) and has the same shape
        as `clicks`.

    Returns:
      A tuple of (exam_given_clicks, rel_given_clicks) representing
      P(examination | click) and P(relevance | click).
    """
    with tf.compat.v1.name_scope(name='compute_latent_prob'):
      is_clicked = tf.greater_equal(tf.cast(clicks, tf.float32), 1.0)
      exam_logits_posterior = exam_logits - tf.math.softplus(rel_logits)
      rel_logits_posterior = rel_logits - tf.math.softplus(exam_logits)
      exam_prob_posterior = tf.compat.v1.where(
          is_clicked, tf.ones_like(exam_logits_posterior),
          tf.sigmoid(exam_logits_posterior))
      rel_prob_posterior = tf.compat.v1.where(
          is_clicked, tf.ones_like(rel_logits_posterior),
          tf.sigmoid(rel_logits_posterior))
      return tf.stop_gradient(exam_prob_posterior), tf.stop_gradient(
          rel_prob_posterior)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """Computes the loss for each element.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size] representing clicks.
      logits: A `Tensor` with shape [batch_size, list_size, 2], where the first
        value in the 3rd-dim is the logits for examination and the second value
        is the logits for relevance.
      mask: A `Tensor` of the same shape as labels indicating which entries are
        valid for computing the loss.

    Returns:
      A tuple(losses, loss_weights).
    """
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    exam_logits, rel_logits = tf.unstack(logits, axis=2)
    exam_logits = tf.compat.v1.where(mask, exam_logits,
                                     tf.zeros_like(exam_logits))
    rel_logits = tf.compat.v1.where(mask, rel_logits, tf.zeros_like(rel_logits))
    # The distribution in the E step.
    exam_latent_prob, rel_latent_prob = self._compute_latent_prob(
        labels, exam_logits, rel_logits)
    # The loss in the M step.
    losses = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
        labels=exam_latent_prob, logits=exam_logits) * self._exam_loss_weight
    losses += tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
        labels=rel_latent_prob, logits=rel_logits) * self._rel_loss_weight
    return losses, tf.cast(mask, dtype=tf.float32)


class SigmoidCrossEntropyLoss(_PointwiseLoss):
  """Implements sigmoid cross entropy loss."""

  def __init__(self, name, temperature=1.0, ragged=False):
    """Overwrite the constructor.

    Args:
      name: A string used as the name for this loss.
      temperature: A float number to modify the logits=logits/temperature.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    super().__init__(name, None, temperature, ragged)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits, tf.zeros_like(logits))
    losses = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return losses, tf.cast(mask, dtype=tf.float32)


class MeanSquaredLoss(_PointwiseLoss):
  """Implements the means squared error loss."""

  def __init__(self, name, ragged=False):
    """Overwrite the constructor.

    Args:
      name: A string used as the name for this loss.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    # temperature is not used in this loss.
    super().__init__(name, None, temperature=1.0, ragged=ragged)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits, tf.zeros_like(logits))
    losses = tf.compat.v1.squared_difference(labels, logits)
    return losses, tf.cast(mask, dtype=tf.float32)


class MixtureEMLoss(_ListwiseLoss):
  """Implements the Mixture EM loss with examination and relevance.

  An Expecatation-Maximization (EM) algorithm is used for estimation and this
  function.
  """

  def __init__(self, name, temperature=1.0, alpha=1.0, ragged=False):
    super().__init__(name, None, temperature, ragged)
    self._alpha = alpha

  def _compute_model_prob(self, per_list_logodds):
    """Computes the probability of models in EM.

    Args:
      per_list_logodds: A `Tensor` with shape [batch_size, 1, model_num].

    Returns:
      A `Tensor` of probability with shape [batch_size, 1, model_num].
    """
    with tf.compat.v1.name_scope(name='compute_model_prob'):
      return tf.stop_gradient(
          tf.exp(-self._alpha *
                 (per_list_logodds -
                  tf.reduce_min(per_list_logodds, axis=2, keepdims=True))))

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """Computes the loss for each element.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size] representing clicks.
      logits: A `Tensor` with shape [batch_size, list_size, model_num], where
        the 3rd-dim is dimension for the models to mix.
      mask: A `Tensor` of the same shape as labels indicating which entries are
        valid for computing the loss.

    Returns:
      A tuple(losses, loss_weights).
    """
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    # The loss in the M step.
    # shape = [batch_size, list_size, model_num]
    losses = tf.stack([
        tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=model_logits)
        for model_logits in tf.unstack(logits, axis=-1)
    ],
                      axis=2)
    losses = tf.where(
        tf.expand_dims(mask, axis=-1), losses,
        tf.zeros_like(losses, dtype=tf.float32))

    # The model probability in the E step.
    losses_no_gradient = tf.stop_gradient(losses)
    # shape = [batch_size, 1, model_num]
    per_list_logodds = tf.reduce_sum(losses_no_gradient, axis=1, keepdims=True)
    model_prob = self._compute_model_prob(per_list_logodds)
    prob_norm = tf.reduce_sum(model_prob, axis=2, keepdims=True)

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(label_sum, 0.0)
    return tf.reshape(
        tf.reduce_sum(losses * model_prob / prob_norm, axis=[1, 2]),
        [-1, 1]), tf.cast(
            nonzero_mask, dtype=tf.float32)


class ListMLELoss(_ListwiseLoss):
  """Implements ListMLE loss."""

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    # Reset the masked labels to 0 and reset the masked logits to a logit with
    # ~= 0 contribution.
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits,
                                tf.math.log(_EPSILON) * tf.ones_like(logits))
    scores = tf.compat.v1.where(
        mask, labels,
        tf.reduce_min(input_tensor=labels, axis=1, keepdims=True) -
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
      batch_size, list_size = tf.unstack(tf.shape(input=sorted_labels))
      sums *= self._lambda_weight.individual_weights(
          sorted_labels,
          tf.tile(tf.expand_dims(tf.range(list_size) + 1, 0), [batch_size, 1]))

    negative_log_likelihood = tf.reduce_sum(
        input_tensor=sums, axis=1, keepdims=True)
    return negative_log_likelihood, tf.ones_like(negative_log_likelihood)


class ApproxNDCGLoss(_ListwiseLoss):
  """Implements ApproxNDCG loss."""

  # Use a different default temperature.
  def __init__(self, name, lambda_weight=None, temperature=0.1, ragged=False):
    """See `_ListwiseLoss`."""
    super().__init__(name, lambda_weight, temperature, ragged)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        mask, logits, -1e3 * tf.ones_like(logits) +
        tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    ranks = approx_ranks(logits)

    return -ndcg(labels, ranks), tf.reshape(
        tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])


class ApproxMRRLoss(_ListwiseLoss):
  """Implements ApproxMRR loss."""

  # Use a different default temperature.
  def __init__(self, name, lambda_weight=None, temperature=0.1, ragged=False):
    """See `_ListwiseLoss`."""
    super().__init__(name, lambda_weight, temperature, ragged)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        mask, logits, -1e3 * tf.ones_like(logits) +
        tf.math.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.math.reduce_sum(input_tensor=labels, axis=1, keepdims=True)

    nonzero_mask = tf.math.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))

    rr = 1. / approx_ranks(logits)
    rr = tf.math.reduce_sum(input_tensor=rr * labels, axis=-1, keepdims=True)
    mrr = rr / tf.math.reduce_sum(input_tensor=labels, axis=-1, keepdims=True)
    return -mrr, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])


class NeuralSortCrossEntropyLoss(_ListwiseLoss):
  """Implements Cross-entropy loss of neural sort permutation matrix."""

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits, tf.zeros_like(logits))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)

    # shape = [batch_size, list_size, list_size].
    true_perm = neural_sort(labels, mask=mask)
    smooth_perm = neural_sort(logits, mask=mask)

    losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=true_perm, logits=tf.math.log(1e-20 + smooth_perm), axis=2)

    # Neural sort will place masked entries last. Losses are still computed on
    # those entries so we need to cancel those out. This means we need to mask
    # out the last n entries, where n is the number of masked items per list. We
    # do so by sorting the mask and setting (masked) invalid losses to 0.
    sorted_mask = tf.cast(
        tf.sort(
            tf.cast(mask, dtype=tf.float32), axis=1, direction='DESCENDING'),
        dtype=tf.bool)
    losses = tf.where(sorted_mask, losses, tf.zeros_like(losses))

    # shape = [batch_size, list_size].
    losses = tf.math.divide_no_nan(
        tf.reduce_sum(input_tensor=losses, axis=-1, keepdims=True),
        tf.reduce_sum(
            input_tensor=tf.cast(mask, dtype=tf.float32),
            axis=-1,
            keepdims=True))

    return losses, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])


class NeuralSortNDCGLoss(_ListwiseLoss):
  """Implements PiRank-NDCG loss.

  The PiRank-NDCG loss is a differentiable approximation of the NDCG metric
  using the NeuralSort trick, which generates a permutation matrix based on
  ranking scores. Please refer to https://arxiv.org/abs/2012.06731 for the
  PiRank method. For PiRank-NDCG in specific,
    NDCG_metric = - sum_i (2^y_i - 1) / log(1 + r_i) / maxDCG,
  where y_i and r_i are the label and the score rank of the ith document
  respectively. This metric can be also written as the sum over rank r with an
  indicator function I,
    NDCG_metric = - sum_{i,r} (2^y_i - 1) / log(1 + r) * I(r, r_i) / maxDCG,
  where the indicator function I(r, r_i) = 1 if r = r_i and 0 otherwise, which
  is the permutation matrix.

  Approximated with a differentiable permutation matrix using neural sort,
    PiRank-NDCG = - sum_{i,r} (2^y_i - 1) / log(1 + r) * P(r, i) / maxDCG,
  where P(r, i) is the approximation of the permutation matrix.
  """

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(mask, logits, tf.zeros_like(logits))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    # shape = [batch_size, list_size].
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    # shape = [batch_size, list_size, list_size].
    smooth_perm = neural_sort(logits, mask=mask)

    return -ndcg(
        labels, perm_mat=smooth_perm), tf.reshape(
            tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])


def neural_sort(logits, name=None, mask=None):
  r"""Generate the permutation matrix from logits by deterministic neuralsort.

  The sort on a list of logits can be approximated by a differentiable
  permutation matrix using Neural Sort (https://arxiv.org/abs/1903.08850).
  The approximation is achieved by constructing a list of functions on logits,
    fn_i(k) = (list_size + 1 - 2*i) * logit_k - sum_j |logit_k - logit_j|,
  whose value is maximal when k is at the ith largest logit.
  So that the permutation matrix can be expressed as
           / 1 if j = argmax_k fn_i(k)
    P_ij = |                           = one_hot(argmax(fn_i(j))).
           \ 0 otherwise
  And the differentiable approximation of the matrix is applied with softmax,
    P^_ij = softmax(fn_i(j) / temperature),
  where the parameter temperature tunes the smoothiness of the approximation.

  #### References
  [1]: Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon.
       Stochastic Optimization of Sorting Networks via Continuous Relaxations.
       https://arxiv.org/abs/1903.08850

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item. (We are using logits here,
      noticing the original paper is using probability weights, i.e., the
      exponentials of the logits).
    name: A string used as the name for this loss.
    mask: A `Tensor` with the same shape as logits indicating which entries are
      valid for computing the neural_sort. Invalid entries are pushed to the
      end.

  Returns:
    A tensor of permutation matrices whose dimension is [batch_size, list_size,
    list_size].
  """
  with tf.compat.v1.name_scope(name, 'neural_sort', [logits]):
    if mask is None:
      mask = tf.ones_like(logits, dtype=tf.bool)

    # Reset logits to 0 and compute number of valid entries for each list in the
    # batch.
    logits = tf.where(mask, logits, tf.zeros_like(logits))
    num_valid_entries = tf.reduce_sum(
        tf.cast(mask, dtype=tf.int32), axis=1, keepdims=True)

    # Compute logit differences and mask out invalid entries.
    logit_diff = tf.abs(tf.expand_dims(logits, 2) - tf.expand_dims(logits, 1))
    valid_pair_mask = _apply_pairwise_op(tf.logical_and, mask)
    logit_diff = tf.where(valid_pair_mask, logit_diff,
                          tf.zeros_like(logit_diff))
    # shape = [batch_size, 1, list_size].
    logit_diff_sum = tf.reduce_sum(
        input_tensor=logit_diff, axis=1, keepdims=True)

    # Compute masked range so that masked items do not influence scaling.
    masked_range = tf.cumsum(tf.cast(mask, dtype=tf.int32), axis=1)
    scaling = tf.cast(
        num_valid_entries + 1 - 2 * masked_range, dtype=tf.float32)
    # shape = [batch_size, list_size].
    scaling = tf.expand_dims(scaling, 2)
    # shape = [batch_size, list_size, list_size].
    # Use broadcast to align the dims.
    scaled_logits = scaling * tf.expand_dims(logits, 1)

    p_logits = scaled_logits - logit_diff_sum

    # Masked entries will be forcefully kept in-place by setting their values to
    # -inf everywhere, except for masked rows where they share equal probability
    # with other masked items.
    p_logits = tf.where(valid_pair_mask, p_logits, -math.inf)
    p_logits = tf.where(
        _apply_pairwise_op(tf.logical_or, mask), p_logits,
        tf.zeros_like(p_logits))

    # By swapping the rows of masked items to the end of the permutation matrix,
    # we force masked items to be placed last.
    sorted_mask_indices = tf.argsort(
        tf.cast(mask, dtype=tf.int32),
        axis=1,
        direction='DESCENDING',
        stable=True)
    p_logits = tf.gather(p_logits, sorted_mask_indices, batch_dims=1, axis=1)

    smooth_perm = tf.nn.softmax(p_logits, -1)

    return smooth_perm


def gumbel_neural_sort(logits,
                       name=None,
                       sample_size=8,
                       temperature=1.0,
                       seed=None):
  """Generate the permutation matrix from logits by stochastic neuralsort.

  By sampling logits from the Gumbel distribution,
    sampled_logits = logits + Gumbel(0, 1),
  the determinstic neural sort z of sampled_logits obeys the distribution with
    Prob(z|logits) = (exp(logit_z1) / Z) * (exp(logit_z2) / Z-exp(logit_z1)) *
                     ... * (exp(logit_zn) / Z-sum_i^(n-1)exp(logit_zi)),
  where Z = sum_i exp(logit_i).

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    name: A string used as the name for this loss.
    sample_size: An integer representing the number of samples drawn from the
      Concrete distribution defined by scores.
    temperature: The Gumbel-Softmax temperature.
    seed: Seed for pseudo-random number generator.

  Returns:
    A `Tensor` of permutation matrices whose dimension is [batch_size,
    sample_size, list_size, list_size].
  """
  with tf.compat.v1.name_scope(name, 'gumbel_neural_sort', [logits]):
    batch_size = tf.shape(input=logits)[0]
    list_size = tf.shape(input=logits)[1]

    # Sample logits from Concrete(logits).
    sampled_logits = tf.expand_dims(logits, 1)
    sampled_logits += _sample_gumbel([batch_size, sample_size, list_size],
                                     seed=seed)
    sampled_logits = tf.reshape(sampled_logits,
                                [batch_size * sample_size, list_size])

    # Sort by constructing the relaxed permuation matrix from sampled logits.
    smooth_perm = neural_sort(sampled_logits / temperature, name)
    smooth_perm = tf.reshape(smooth_perm,
                             [batch_size, sample_size, list_size, list_size])

    return smooth_perm


class OrdinalLoss(_PointwiseLoss):
  """Implements ordinal loss."""

  def __init__(self, name, ordinal_size, temperature=1.0, ragged=False,
               use_fraction_label=False):
    """Initializer.

    Args:
      name: A string used as the name for this loss.
      ordinal_size: A integer number of ordinal levels of labels.
      temperature: A float number to modify the logits=logits/temperature.
      ragged: A boolean indicating whether the input tensors are ragged.
      use_fraction_label: A boolean indicating when the input labels contain
        fractions, whether to leverage the fraction part.
    """
    super().__init__(name, None, temperature, ragged)
    self._ordinal_size = ordinal_size
    self._use_fraction_label = use_fraction_label

  def _labels_to_ordinals(self, labels, mask):
    """Helper function to transform input labels to ordinal values.

    When use_fraction_label is false, ordinals will be 1.0 if labels >= i for
    the ordinal head i, with i = 1, ..., ordinal_size.
    When use_fraction_label is true, the fraction part of labels will be counted
    if labels > i-1 but < i.

    For a fraction label 1.2, and ordinal_size=2
    when use_fraction_label is false, it maps to an ordinal like [1.0, 0.0],
    when use_fraction_label is true, it maps to an ordinal like [1.0, 0.2].

    Args:
      labels: A Tensor of shape [batch_size, list_size].
      mask: A Tensor of shape [batch_size, list_size].

    Returns:
      ordinals, shape [batch_size, list_size, ordinal_size]
    """
    one_to_n = tf.range(1, self._ordinal_size + 1, dtype=tf.float32)
    unsqueezed = tf.repeat(
        tf.expand_dims(labels, axis=2), self._ordinal_size, axis=-1)
    ordinals = tf.where(unsqueezed >= one_to_n, tf.ones_like(unsqueezed), 0.0)
    if self._use_fraction_label:
      fractions = unsqueezed - one_to_n + 1.0
      fractions = tf.where(
          tf.logical_and(fractions > 0.0, fractions < 1.0), fractions, 0.0)
      ordinals += fractions
    return tf.where(tf.expand_dims(mask, axis=-1), ordinals, 0.0)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    if logits.shape.rank != 3:
      raise ValueError('Predictions for ordinal loss must have rank 3.')
    elif logits.shape[-1] != self._ordinal_size:
      raise ValueError(
          'The last dimension of logits must be the number of ordinal levels '
          f'{self._ordinal_size}, the actual dimension is {logits.shape[-1]}.')
    labels = tf.where(mask, labels, 0.0)
    logits = tf.where(tf.expand_dims(mask, -1), logits, 0.0)
    ordinals = self._labels_to_ordinals(labels, mask)
    losses = tf.where(
        tf.expand_dims(mask, -1),
        tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            labels=ordinals,
            logits=logits),
        0.0)
    return tf.reduce_sum(losses, axis=-1), tf.cast(mask, dtype=tf.float32)


class MultiClassLoss(_PointwiseLoss):
  """Implements multi-class loss."""

  def __init__(self,
               name,
               num_classes,
               temperature=1.0,
               ragged=False,
               from_logits=False,
               label_smoothing=0.0):
    """Initializer.

    Args:
      name: A string used as the name for this loss.
      num_classes: A integer number of classes. To use this loss,
        num_classes must be greater than 1.
      temperature: A float number to modify the logits=logits/temperature.
      ragged: A boolean indicating whether the input tensors are ragged.
      from_logits: A boolean indicating whether the input is logits or probs.
      label_smoothing: A float number of label smoothing.
    """
    super().__init__(name, None, temperature, ragged)
    self._num_classes = num_classes
    self._from_logits = from_logits
    self._label_smoothing = label_smoothing

  def _labels_to_one_hot_class(self, labels, mask):
    """Helper function to transform input labels to one hot class labels.

    Args:
      labels: A Tensor of shape [batch_size, list_size].
      mask: A Tensor of shape [batch_size, list_size].

    Returns:
      one-hot class label, shape [batch_size, list_size, num_classes]
    """
    classes = tf.one_hot(
        tf.cast(labels, tf.int32), self._num_classes, dtype=tf.float32)
    return tf.where(tf.expand_dims(mask, axis=-1), classes, 0.0)

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    if logits.shape.rank != 3:
      raise ValueError('Predictions for multi-class loss must have rank 3.')
    elif logits.shape[-1] != self._num_classes:
      raise ValueError(
          'The last dimension of logits must be the number of classes '
          f'{self._num_classes}, the actual dimension is {logits.shape[-1]}.')
    labels = tf.where(mask, labels, 0.0)
    logits = tf.where(tf.expand_dims(mask, -1), logits, 0.0)
    classes = self._labels_to_one_hot_class(labels, mask)
    losses = tf.keras.losses.CategoricalCrossentropy(
        from_logits=self._from_logits,
        label_smoothing=self._label_smoothing,
        axis=-1,
        reduction=tf.keras.losses.Reduction.NONE,
        name='categorical_crossentropy')(
            classes, logits, tf.cast(mask, dtype=tf.float32))
    return losses, tf.cast(mask, dtype=tf.float32)


class CoupledRankDistilLoss(_ListwiseLoss):
  r"""Implements Coupled-RankDistil loss.

  The Coupled-RankDistil loss ([Reddi et al, 2021][reddi2021]) is the
  cross-entropy between k-Plackett's probability of logits (student) and labels
  (teacher).

  The k-Plackett's probability model is defined as:
  $$
  \mathcal{P}_k(\pi|s) = \frac{1}{(N-k)!} \\
  \frac{\prod_{i=1}^k exp(s_{\pi(i)})}{\sum_{j=k}^N log(exp(s_{\pi(i)}))}.
  $$

  The Coupled-RankDistil loss is defined as:
  $$
  \mathcal{L}(y, s) = -\sum_{\pi} \mathcal{P}_k(\pi|y) log\mathcal{P}(\pi|s) \\
  =  \mathcal{E}_{\pi \sim \matcal{P}(.|y)} [-\log \mathcal{P}(\pi|s)]
  $$

    References:
    - [RankDistil: Knowledge Distillation for Ranking, Reddi et al,
       2021][reddi2021]

  [reddi2021]: https://research.google/pubs/pub50695/
  """

  def __init__(self,
               name,
               sample_size,
               topk=None,
               temperature=1.,
               ragged=False):
    """Initializer.

    Args:
      name: A string used as the name for this loss.
      sample_size: Number of permutations to sample from teacher scores.
      topk: top-k entries over which order is matched. A penalty is applied over
        non top-k items.
      temperature: A float number to modify the logits as
        `logits=logits/temperature`.
      ragged: A boolean indicating whether the input tensors are ragged.
    """
    super().__init__(name, None, temperature, ragged)
    self._sample_size = sample_size
    self._topk = topk

  def _compute_unreduced_loss_impl(self, labels, logits, mask=None):
    """See `_RankingLoss`."""
    if mask is None:
      mask = utils.is_label_valid(labels)
    labels = tf.where(mask, labels, tf.zeros_like(labels))
    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)

    teacher_scores = tf.where(mask, labels,
                              tf.math.log(_EPSILON) * tf.ones_like(labels))

    student_scores = tf.where(mask, logits,
                              tf.math.log(_EPSILON) * tf.ones_like(logits))

    # Sample teacher scores.
    # [batch_size, list_size] -> [batch_size, sample_size, list_size].
    sampled_teacher_scores = tf.expand_dims(teacher_scores, 1)
    sampled_teacher_scores = tf.repeat(
        sampled_teacher_scores, [self._sample_size], axis=1)

    batch_size, list_size = tf.unstack(tf.shape(input=labels))
    sampled_teacher_scores += _sample_gumbel(
        [batch_size, self._sample_size, list_size], seed=37)
    sampled_teacher_scores = tf.math.log(
        tf.nn.softmax(sampled_teacher_scores) + _EPSILON)

    # Expand student scores.
    # [batch_size, list_size] -> [batch_size, sample_size, list_size].
    expanded_student_scores = tf.expand_dims(student_scores, 1)
    expanded_student_scores = tf.repeat(
        expanded_student_scores, [self._sample_size], axis=1)

    # Sort teacher scores and student scores to obtain top-k student scores
    # whose order is based on teacher scores.
    sorted_student_scores = utils.sort_by_scores(
        utils.reshape_first_ndims(sampled_teacher_scores, 2,
                                  [batch_size * self._sample_size]),
        [
            utils.reshape_first_ndims(expanded_student_scores, 2,
                                      [batch_size * self._sample_size])
        ],
        shuffle_ties=True,
        seed=37)[0]
    sorted_student_scores = utils.reshape_first_ndims(
        sorted_student_scores, 1, [batch_size, self._sample_size])
    topk = self._topk or list_size
    topk_student_scores = sorted_student_scores[:, :, :topk]

    # For \pi from teacher scores, compute top-k Plackett's probability as:
    # \prod_{i=1}^k exp(s_{\pi(i)}) / \sum_{j=k}^N log(exp(s_{\pi(i)})).

    # Compute the denominator mask for  \sum_{j=k}^N log(exp(s_{\pi(i)}).
    # We apply logsumexp over valid entries in this mask.
    # topk_pl_denominator_mask = batch x sample_size x valid_denom_entries,
    # where valid_denom_entries = [[1 1 1 1 1 1]
    #                             [0 1 1 1 1 1]
    #                             [0 0 1 1 1 1]].
    # An alternative implementation would be to use `cumulative_logsumexp` with
    # `reverse=True` to compute the denominator term.
    ones = tf.ones((topk, list_size), dtype=tf.float32)
    ones_upper = tf.linalg.band_part(ones, 0, -1)
    topk_pl_denominator_mask = tf.tile(
        tf.expand_dims(ones_upper, axis=0),
        [batch_size * self._sample_size, 1, 1])
    # [batch_size * sample_size, topk, list_size] ->
    # [batch_size, sample_size, topk, list_size].
    topk_pl_denominator_mask = tf.cast(
        utils.reshape_first_ndims(topk_pl_denominator_mask, 1,
                                  [batch_size, self._sample_size]),
        dtype=tf.bool)
    sorted_student_scores = tf.tile(
        tf.expand_dims(sorted_student_scores, 2), [1, 1, topk, 1])

    sorted_student_scores_denom = tf.where(
        topk_pl_denominator_mask, sorted_student_scores,
        tf.math.log(_EPSILON) * tf.ones_like(sorted_student_scores))
    logprob = topk_student_scores - tf.math.reduce_logsumexp(
        sorted_student_scores_denom, axis=3)
    # Compute log-likelihood over top-k Plackett-Luce scores.
    # [batch_size, sample_size, topk] -> [batch_size, sample_size].
    logprob = tf.reduce_sum(logprob, axis=2)

    # Compute RankDistil loss as a mean over samples.
    # [batch_size, sample_size] -> [batch_size, 1].
    nll = tf.reduce_mean(-logprob, axis=1, keepdims=True)

    return nll, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])
