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

"""Utility functions for ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops


def is_label_valid(labels):
  """Returns a boolean `Tensor` for label validity."""
  labels = ops.convert_to_tensor(labels)
  return math_ops.greater_equal(labels, 0.)


def sort_by_scores(scores, features_list, topn=None):
  """Sorts example features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s with the same shape as scores to be
      sorted.
    topn: An integer as the cutoff of examples in the sorted list.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  scores = ops.convert_to_tensor(scores)
  scores.get_shape().assert_has_rank(2)
  batch_size, list_size = array_ops.unstack(array_ops.shape(scores))
  if topn is None:
    topn = list_size
  topn = math_ops.minimum(topn, list_size)
  _, indices = nn_ops.top_k(scores, topn, sorted=True)
  list_offsets = array_ops.expand_dims(
      math_ops.range(batch_size) * list_size, 1)
  # The shape of `indices` is [batch_size, topn] and the shape of
  # `list_offsets` is [batch_size, 1]. Broadcasting is used here.
  gather_indices = array_ops.reshape(indices + list_offsets, [-1])
  output_shape = array_ops.stack([batch_size, topn])
  # Each feature is first flattened to a 1-D vector and then gathered by the
  # indices from sorted scores and then re-shaped.
  return [
      array_ops.reshape(
          array_ops.gather(array_ops.reshape(feature, [-1]), gather_indices),
          output_shape) for feature in features_list
  ]


def shuffle_valid_indices(is_valid, seed=None):
  """Returns a shuffle of indices with valid ones on top."""
  return organize_valid_indices(is_valid, shuffle=True, seed=seed)


def organize_valid_indices(is_valid, shuffle=True, seed=None):
  """Organizes indices in such a way that valid items appear first.

  Args:
    is_valid: A boolen `Tensor` for entry validity with shape [batch_size,
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
  is_valid = ops.convert_to_tensor(is_valid)
  is_valid.get_shape().assert_has_rank(2)
  output_shape = array_ops.shape(is_valid)

  if shuffle:
    values = random_ops.random_uniform(output_shape, seed=seed)
  else:
    values = (array_ops.ones_like(is_valid, dtypes.float32) *
              array_ops.reverse(
                  math_ops.to_float(math_ops.range(output_shape[1])), [-1]))

  rand = array_ops.where(
      is_valid, values, array_ops.ones(output_shape) * -1e-6)
  # shape(indices) = [batch_size, list_size]
  _, indices = nn_ops.top_k(rand, output_shape[1], sorted=True)
  # shape(batch_ids) = [batch_size, list_size]
  batch_ids = array_ops.ones_like(indices) * array_ops.expand_dims(
      math_ops.range(output_shape[0]), 1)
  return array_ops.concat(
      [array_ops.expand_dims(batch_ids, 2),
       array_ops.expand_dims(indices, 2)],
      axis=2)


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
  new_shape = array_ops.concat(
      [new_shape, array_ops.shape(tensor)[first_ndims:]], 0)
  if isinstance(tensor, sparse_tensor.SparseTensor):
    return sparse_ops.sparse_reshape(tensor, new_shape)

  return array_ops.reshape(tensor, new_shape)


def approx_ranks(logits, alpha=10.):
  r"""Computes approximate ranks given a list of logits.

  Given a list of logits, the rank of an item in the list is simply
  one plus the total number of items with a larger logit. In other words,

    rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

  where "I" is the indicator function. The indicator function can be
  approximated by a generalized sigmoid:

    I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).

  This function approximates the rank of an item using this sigmoid
  approximation to the indicator function. This technique is at the core
  of "A general approximation framework for direct optimization of
  information retrieval measures" by Qin et al.

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    alpha: Exponent of the generalized sigmoid function.

  Returns:
    A `Tensor` of ranks with the same shape as logits.
  """
  list_size = array_ops.shape(logits)[1]
  x = array_ops.tile(
      array_ops.expand_dims(logits, 2), [1, 1, list_size])
  y = array_ops.tile(
      array_ops.expand_dims(logits, 1), [1, list_size, 1])
  pairs = math_ops.sigmoid(alpha * (y - x))
  return math_ops.reduce_sum(pairs, -1) + .5


def inverse_max_dcg(labels,
                    gain_fn=lambda labels: math_ops.pow(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / math_ops.log1p(rank),
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
  ideal_sorted_labels, = sort_by_scores(labels, [labels], topn=topn)
  rank = math_ops.range(array_ops.shape(ideal_sorted_labels)[1]) + 1
  discounted_gain = gain_fn(
      ideal_sorted_labels) * rank_discount_fn(math_ops.to_float(rank))
  discounted_gain = math_ops.reduce_sum(discounted_gain, 1, keepdims=True)
  return array_ops.where(
      math_ops.greater(discounted_gain, 0.), 1. / discounted_gain,
      array_ops.zeros_like(discounted_gain))
