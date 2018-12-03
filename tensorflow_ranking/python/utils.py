# Copyright 2018 The TensorFlow Ranking Authors.
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
  """Returns a shuffle of indices with valid ones on top.

  Args:
    is_valid: A boolen `Tensor` for entry validity with shape [batch_size,
      list_size].
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
  rand = array_ops.where(is_valid,
                         random_ops.random_uniform(output_shape, seed=seed),
                         array_ops.ones(output_shape) * -1e-6)
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
