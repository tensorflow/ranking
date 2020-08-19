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

"""Utility functions for ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _to_nd_indices(indices):
  """Returns indices used for tf.gather_nd or tf.scatter_nd.

  Args:
    indices: A `Tensor` of shape [batch_size, size] with integer values. The
      values are the indices of another `Tensor`. For example, `indices` is the
      output of tf.argsort or tf.math.top_k.

  Returns:
    A `Tensor` with shape [batch_size, size, 2] that can be used by tf.gather_nd
    or tf.scatter_nd.

  """
  indices.get_shape().assert_has_rank(2)
  batch_ids = tf.ones_like(indices) * tf.expand_dims(
      tf.range(tf.shape(input=indices)[0]), 1)
  return tf.stack([batch_ids, indices], axis=-1)


def is_label_valid(labels):
  """Returns a boolean `Tensor` for label validity."""
  labels = tf.convert_to_tensor(value=labels)
  return tf.greater_equal(labels, 0.)


def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None):
  """Sorts list of features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s to be sorted. The shape of the `Tensor`
      can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
      The latter is applicable for example features.
    topn: An integer as the cutoff of examples in the sorted list.
    shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    seed: The ops-level random seed used when `shuffle_ties` is True.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  with tf.compat.v1.name_scope(name='sort_by_scores'):
    scores = tf.cast(scores, tf.float32)
    scores.get_shape().assert_has_rank(2)
    list_size = tf.shape(input=scores)[1]
    if topn is None:
      topn = list_size
    topn = tf.minimum(topn, list_size)
    shuffle_ind = None
    if shuffle_ties:
      shuffle_ind = _to_nd_indices(
          tf.argsort(
              tf.random.uniform(tf.shape(input=scores), seed=seed),
              stable=True))
      scores = tf.gather_nd(scores, shuffle_ind)
    _, indices = tf.math.top_k(scores, topn, sorted=True)
    nd_indices = _to_nd_indices(indices)
    if shuffle_ind is not None:
      nd_indices = tf.gather_nd(shuffle_ind, nd_indices)
    return [tf.gather_nd(f, nd_indices) for f in features_list]


def sorted_ranks(scores, shuffle_ties=True, seed=None):
  """Returns an int `Tensor` as the ranks (1-based) after sorting scores.

  Example: Given scores = [[1.0, 3.5, 2.1]], the returned ranks will be [[3, 1,
  2]]. It means that scores 1.0 will be ranked at position 3, 3.5 will be ranked
  at position 1, and 2.1 will be ranked at position 2.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    shuffle_ties: See `sort_by_scores`.
    seed: See `sort_by_scores`.

  Returns:
    A 1-based int `Tensor`s as the ranks.
  """
  with tf.compat.v1.name_scope(name='sorted_ranks'):
    batch_size, list_size = tf.unstack(tf.shape(input=scores))
    # The current position in the list for each score.
    positions = tf.tile(tf.expand_dims(tf.range(list_size), 0), [batch_size, 1])
    # For score [[1.0, 3.5, 2.1]], sorted_positions are [[1, 2, 0]], meaning the
    # largest score is at position 1, the 2nd is at position 2 and 3rd is at
    # position 0.
    sorted_positions = sort_by_scores(
        scores, [positions], shuffle_ties=shuffle_ties, seed=seed)[0]
    # The indices of sorting sorted_positions will be [[2, 0, 1]] and ranks are
    # 1-based and thus are [[3, 1, 2]].
    ranks = tf.argsort(sorted_positions) + 1
    return ranks


def shuffle_valid_indices(is_valid, seed=None):
  """Returns a shuffle of indices with valid ones on top."""
  return organize_valid_indices(is_valid, shuffle=True, seed=seed)


def organize_valid_indices(is_valid, shuffle=True, seed=None):
  """Organizes indices in such a way that valid items appear first.

  Args:
    is_valid: A boolean `Tensor` for entry validity with shape [batch_size,
      list_size].
    shuffle: A boolean indicating whether valid items should be shuffled.
    seed: An int for random seed at the op level. It works together with the
      seed at global graph level together to determine the random number
      generation. See `tf.set_random_seed`.

  Returns:
    A tensor of indices with shape [batch_size, list_size, 2]. The returned
    tensor can be used with `tf.gather_nd` and `tf.scatter_nd` to compose a new
    [batch_size, list_size] tensor. The values in the last dimension are the
    indices for an element in the input tensor.
  """
  with tf.compat.v1.name_scope(name='organize_valid_indices'):
    is_valid = tf.convert_to_tensor(value=is_valid)
    is_valid.get_shape().assert_has_rank(2)
    output_shape = tf.shape(input=is_valid)

    if shuffle:
      values = tf.random.uniform(output_shape, seed=seed)
    else:
      values = (
          tf.ones_like(is_valid, tf.float32) * tf.reverse(
              tf.cast(tf.range(output_shape[1]), dtype=tf.float32), [-1]))

    rand = tf.compat.v1.where(is_valid, values, tf.ones(output_shape) * -1e-6)
    # shape(indices) = [batch_size, list_size]
    indices = tf.argsort(rand, direction='DESCENDING', stable=True)
    return _to_nd_indices(indices)


def reshape_first_ndims(tensor, first_ndims, new_shape):
  """Reshapes the first n dims of the input `tensor` to `new shape`.

  Args:
    tensor: The input `Tensor`.
    first_ndims: A int denoting the first n dims.
    new_shape: A list of int representing the new shape.

  Returns:
    A reshaped `Tensor`.
  """
  assert tensor.get_shape().ndims is None or tensor.get_shape(
  ).ndims >= first_ndims, (
      'Tensor shape is less than {} dims.'.format(first_ndims))
  new_shape = tf.concat([new_shape, tf.shape(input=tensor)[first_ndims:]], 0)
  if isinstance(tensor, tf.SparseTensor):
    return tf.sparse.reshape(tensor, new_shape)

  return tf.reshape(tensor, new_shape)


def reshape_to_2d(tensor):
  """Converts the given `tensor` to a 2-D `Tensor`."""
  with tf.compat.v1.name_scope(name='reshape_to_2d'):
    rank = tensor.shape.rank if tensor.shape is not None else None
    if rank is not None and rank != 2:
      if rank >= 3:
        tensor = tf.reshape(tensor, tf.shape(input=tensor)[0:2])
      else:
        while tensor.shape.rank < 2:
          tensor = tf.expand_dims(tensor, -1)
    return tensor


def _circular_indices(size, num_valid_entries):
  """Creates circular indices with padding and mask for non-padded ones.

  This returns a indices and a mask Tensor, where the mask is True for valid
  entries and False for padded entries.

  The returned indices have the shape of [batch_size, size], where the
  batch_size is obtained from the 1st dim of `num_valid_entries`. For a
  batch_size = 1, when size = 3, returns [[0, 1, 2]], when num_valid_entries =
  2, returns [[0, 1, 0]]. The first 2 are valid and the returned mask is [True,
  True, False].

  Args:
    size: A scalar int `Tensor` for the size.
    num_valid_entries: A 1-D `Tensor` with shape [batch_size] representing the
      number of valid entries for each instance in a batch.

  Returns:
    A tuple of Tensors (batch_indices, batch_indices_mask). The first has
    shape [batch_size, size] and the second has shape [batch_size, size].
  """
  with tf.compat.v1.name_scope(name='circular_indices'):
    # shape = [batch_size, size] with value [[0, 1, ...], [0, 1, ...], ...].
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(size), 0),
        [tf.shape(input=num_valid_entries)[0], 1])
    num_valid_entries = tf.reshape(num_valid_entries, [-1, 1])
    batch_indices_mask = tf.less(batch_indices, num_valid_entries)
    # Use mod to make the indices to the ranges of valid entries.
    num_valid_entries = tf.compat.v1.where(
        tf.less(num_valid_entries, 1), tf.ones_like(num_valid_entries),
        num_valid_entries)
    batch_indices = tf.math.mod(batch_indices, num_valid_entries)
    return batch_indices, batch_indices_mask


def padded_nd_indices(is_valid, shuffle=False, seed=None):
  """Pads the invalid entries by valid ones and returns the nd_indices.

  For example, when we have a batch_size = 1 and list_size = 3. Only the first 2
  entries are valid. We have:
  ```
  is_valid = [[True, True, False]]
  nd_indices, mask = padded_nd_indices(is_valid)
  ```
  nd_indices has a shape [1, 3, 2] and mask has a shape [1, 3].

  ```
  nd_indices = [[[0, 0], [0, 1], [0, 0]]]
  mask = [[True, True, False]]
  ```
  nd_indices can be used by gather_nd on a Tensor t
  ```
  padded_t = tf.gather_nd(t, nd_indices)
  ```
  and get the following Tensor with first 2 dims are [1, 3]:
  ```
  padded_t = [[t(0, 0), t(0, 1), t(0, 0)]]
  ```

  Args:
    is_valid: A boolean `Tensor` for entry validity with shape [batch_size,
      list_size].
    shuffle: A boolean that indicates whether valid indices should be shuffled.
    seed: Random seed for shuffle.

  Returns:
    A tuple of Tensors (nd_indices, mask). The first has shape [batch_size,
    list_size, 2] and it can be used in gather_nd or scatter_nd. The second has
    the shape of [batch_size, list_size] with value True for valid indices.
  """
  with tf.compat.v1.name_scope(name='nd_indices_with_padding'):
    is_valid = tf.convert_to_tensor(value=is_valid)
    list_size = tf.shape(input=is_valid)[1]
    num_valid_entries = tf.reduce_sum(
        input_tensor=tf.cast(is_valid, dtype=tf.int32), axis=1)
    indices, mask = _circular_indices(list_size, num_valid_entries)
    # Valid indices of the tensor are shuffled and put on the top.
    # [batch_size, list_size, 2].
    shuffled_indices = organize_valid_indices(
        is_valid, shuffle=shuffle, seed=seed)
    # Construct indices for gather_nd [batch_size, list_size, 2].
    nd_indices = _to_nd_indices(indices)
    nd_indices = tf.gather_nd(shuffled_indices, nd_indices)
    return nd_indices, mask


def de_noise(counts, noise, ratio=0.9):
  """Returns a float `Tensor` as the de-noised `counts`.

  The implementation is based on the the paper by Zhang and Xu: "Fast Exact
  Maximum Likelihood Estimation for Mixture of Language Models." It assumes that
  the observed `counts` are generated from a mixture of `noise` and the true
  distribution: `ratio * noise_distribution + (1 - ratio) * true_distribution`,
  where the contribution of `noise` is controlled by `ratio`. This method
  returns the true distribution.

  Args:
    counts: A 2-D `Tensor` representing the observations. All values should be
      nonnegative.
    noise: A 2-D `Tensor` representing the noise distribution. This should be
      the same shape as `counts`. All values should be positive and are
      normalized to a simplex per row.
    ratio: A float in (0, 1) representing the contribution from noise.

  Returns:
    A 2-D float `Tensor` and each row is a simplex.
  Raises:
    ValueError: if `ratio` is not in (0,1).
    InvalidArgumentError: if any of `counts` is negative or any of `noise` is
    not positive.
  """
  if not 0 < ratio < 1:
    raise ValueError('ratio should be in (0, 1), but get {}'.format(ratio))
  odds = (1 - ratio) / ratio

  counts = tf.cast(counts, dtype=tf.float32)
  noise = tf.cast(noise, dtype=tf.float32)

  counts.get_shape().assert_has_rank(2)
  noise.get_shape().assert_has_rank(2)
  noise.get_shape().assert_is_compatible_with(counts.get_shape())

  with tf.compat.v1.name_scope(name='de_noise'):
    counts_nonneg = tf.compat.v1.assert_greater_equal(counts, 0.)
    noise_pos = tf.compat.v1.assert_greater(noise, 0.)
    with tf.control_dependencies([counts_nonneg, noise_pos]):
      # Normalize noise to be a simplex per row.
      noise = noise / tf.reduce_sum(noise, axis=1, keepdims=True)
      sorted_idx = tf.argsort(
          counts / noise, direction='DESCENDING', stable=True)
      nd_indices = _to_nd_indices(sorted_idx)
      sorted_counts = tf.gather_nd(counts, nd_indices)
      sorted_noise = tf.gather_nd(noise, nd_indices)
      # Decide whether an entry will have a positive value or 0.
      is_pos = tf.cast(
          (odds + tf.cumsum(sorted_noise, axis=1)) /
          tf.cumsum(sorted_counts, axis=1) > sorted_noise / sorted_counts,
          tf.float32)
      # The lambda in the paper above, which is the lagrangian multiplier for
      # the simplex constraint on the variables.
      lagrangian_multiplier = tf.reduce_sum(
          sorted_counts * is_pos, axis=1, keepdims=True) / (1 + tf.reduce_sum(
              sorted_noise * is_pos, axis=1, keepdims=True) / odds)
      res = (sorted_counts / lagrangian_multiplier -
             sorted_noise / odds) * is_pos
      return tf.scatter_nd(nd_indices, res, shape=tf.shape(counts))
