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

"""Tensorflow Ranking model library.

This provides functions to build `model_fn` used in `tf.estimator`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import function_utils
from tensorflow.python.estimator import model_fn

from tensorflow_ranking.python import feature
from tensorflow_ranking.python import utils


def _rolling_window_indices(size, rw_size, num_valid_entries):
  """Returns the rolling windows indices and mask for valid ones.

  When size = 3, rw_size = 2, returns [[0, 1], [1, 2], [2, 0]]. When size = 2,
  rw_size = 3, returns [[0, 1, 0], [1, 0, 1]].

  When num_valid_entries = 2, the first returns [[0, 1], [1, 0], [0, 1]] and the
  first 2 are valid with mask as [True, True, False].

  Args:
    size: A scalar int `Tensor` for the size.
    rw_size: A scalr int `Tensor` for the rw_size.
    num_valid_entries: A 1-D `Tensor` with shape [batch_size] representing the
      number of valid entries for each instance in a batch.

  Returns:
    A tuple of Tensors (batch_rw_indices, batch_indices_mask). The first has
    shape [batch_size, size, rw_size] and the second has shape [batch_size,
    size].
  """
  with ops.name_scope(None, 'rolling_window_indices',
                      (size, rw_size, num_valid_entries)):
    # shape = [size, rw_size] with value [[0, 1, ...], [1, 2, ...], ...].
    rw_indices = array_ops.expand_dims(math_ops.range(rw_size),
                                       0) + array_ops.expand_dims(
                                           math_ops.range(size), 1)
    # shape = [batch_size, size, rw_size]. Make batch_size copies.
    batch_rw_indices = array_ops.gather(
        array_ops.expand_dims(rw_indices, 0),
        array_ops.zeros_like(num_valid_entries),
        axis=0)
    # Mark the first n indices as valid where n = num_valid_entries.
    batch_indices_mask = math_ops.less(
        math_ops.reduce_min(batch_rw_indices, axis=2),
        array_ops.reshape(num_valid_entries, [-1, 1]))
    # Mod the indices to the range of num_valid_entries.
    num_valid_entries = array_ops.where(
        math_ops.less(num_valid_entries, 1),
        array_ops.ones_like(num_valid_entries), num_valid_entries)
    batch_rw_indices = math_ops.mod(
        batch_rw_indices, array_ops.reshape(num_valid_entries, [-1, 1, 1]))
    return batch_rw_indices, batch_indices_mask


def _form_group_indices_nd(is_valid, group_size, shuffle=True):
  """Forms the indices for groups for gather_nd or scatter_nd.

  Args:
    is_valid: A boolen `Tensor` for entry validity with shape [batch_size,
      list_size].
    group_size: An scalar int `Tensor` for the number of examples in a group.
    shuffle: A boolean that indicates whether valid indices should be shuffled
      when forming group indices.

  Returns:
    A tuple of Tensors (indices, mask). The first has shape [batch_size,
    num_groups, group_size, 2] and it can be used in gather_nd or scatter_nd for
    group features. The second has the shape of [batch_size, num_groups] with
    value True for valid groups.
  """
  with ops.name_scope(None, 'form_group_indices', (is_valid, group_size)):
    is_valid = ops.convert_to_tensor(is_valid)
    batch_size, list_size = array_ops.unstack(array_ops.shape(is_valid))
    num_valid_entries = math_ops.reduce_sum(math_ops.to_int32(is_valid), axis=1)
    rw_indices, mask = _rolling_window_indices(list_size, group_size,
                                               num_valid_entries)
    # Valid indices of the tensor are shuffled and put on the top.
    # [batch_size, list_size, 2]. A determinstic op-level seed is set mainly for
    # unittest purpose. We can find a better way to avoid setting this seed
    # explicitly.
    shuffled_indices = utils.organize_valid_indices(
        is_valid, shuffle=shuffle, seed=87124)
    # Construct indices for gather_nd.
    # [batch_size, num_groups, group_size, 2]
    group_indices_nd = array_ops.expand_dims(rw_indices, axis=3)
    group_indices_nd = array_ops.concat([
        array_ops.reshape(math_ops.range(batch_size), [-1, 1, 1, 1]) *
        array_ops.ones_like(group_indices_nd), group_indices_nd
    ], 3)

    indices = array_ops.gather_nd(shuffled_indices, group_indices_nd)
    return indices, mask


def make_groupwise_ranking_fn(group_score_fn,
                              group_size,
                              ranking_head,
                              transform_fn=None):
  """Builds an `Estimator` model_fn for groupwise comparison ranking models.

  Args:
    group_score_fn: Scoring function for a group of examples with `group_size`
      that returns a score per example. It has to follow signature:
      * Args:
        `context_features`: A dict of `Tensor`s with shape [batch_size, ...].
        `per_example_features`: A dict of `Tensor`s with shape [batch_size,
          group_size, ...]
        `mode`: Optional. Specifies if this is training, evaluation or
          inference. See `ModeKeys`.
        `params`: Optional dict of hyperparameters, same value passed in the
          `Estimator` constructor.
        `config`: Optional configuration object, same value passed in the
          `Estimator` constructor.
      * Returns: Tensor of shape [batch_size, group_size] containing per-example
        scores.
    group_size: An integer denoting the number of examples in `group_score_fn`.
    ranking_head: A `head._RankingHead` object.
    transform_fn: Function transforming the raw features into dense tensors. It
      has the following signature:
      * Args:
        `features`: A dict of `Tensor`s contains the raw input.
        `mode`: Optional. See estimator `ModeKeys`.
      * Returns:
        `context_features`: A dict of `Tensor`s with shape [batch_size, ...]
        `per_example_features`: A dict of `Tensor`s with shape [batch_size,
          list_size, ...]

  Returns:
    An `Estimator` `model_fn` (see estimator.py) with the following signature:
    * Args:
      * `features`: dict of Tensors of shape [batch_size, list_size, ...] for
      per-example features and shape [batch_size, ...] for non-example context
      features.
      * `labels`: Tensor with shape [batch_size, list_size] denoting relevance.
      * `mode`: No difference.
      * `params`: No difference.
      * `config`: No difference..
    * Returns:
      `EstimatorSpec`
  Raises:
    ValueError: when group_size is invalid.
  """
  if group_size <= 0:
    raise ValueError('Invalid group_size %d' % group_size)
  if transform_fn is None:
    transform_fn = feature.make_identity_transform_fn({})

  def _call_transform_fn(features, mode):
    """Calling transform function."""
    transform_fn_args = function_utils.fn_args(transform_fn)
    if 'mode' in transform_fn_args:
      return transform_fn(features, mode=mode)
    else:
      return transform_fn(features)

  def _groupwise_dnn_v2(features, labels, mode, params, config):
    """Defines the dnn for groupwise scoring functions."""
    with ops.name_scope('transform'):
      context_features, per_example_features = _call_transform_fn(
          features, mode)

    def _score_fn(context_features, group_features, reuse):
      with variable_scope.variable_scope('group_score', reuse=reuse):
        return group_score_fn(context_features, group_features, mode, params,
                              config)

    # Scatter/Gather per-example scores through groupwise comparison. Each
    # instance in a mini-batch will form a number of groups. Each groups of
    # examples are scored by 'score_fn' and socres for individual examples
    # accumulated over groups.
    with ops.name_scope('groupwise_dnn_v2'):
      with ops.name_scope('infer_sizes'):
        if labels is not None:
          batch_size, list_size = array_ops.unstack(array_ops.shape(labels))
          is_valid = utils.is_label_valid(labels)
        else:
          # Infer batch_size and list_size from a feature.
          example_tensor_shape = array_ops.shape(
              next(six.itervalues(per_example_features)))
          batch_size = example_tensor_shape[0]
          list_size = example_tensor_shape[1]
          is_valid = utils.is_label_valid(
              array_ops.ones([batch_size, list_size]))
      if batch_size is None or list_size is None:
        raise ValueError(
            'Invalid batch_size=%s or list_size=%s' % (batch_size, list_size))

      # For each example feature, assume the shape is [batch_size, list_size,
      # feature_size], the groups are formed along the 2nd dim. Each group has a
      # 'group_size' number of indices in [0, list_size). Based on these
      # indices, we can gather the example feature into a sub-tensor for each
      # group. The total number of groups we have for a mini-batch is batch_size
      # * num_groups. Inside each group, we have a 'group_size' number of
      # examples.
      indices, mask = _form_group_indices_nd(
          is_valid, group_size,
          shuffle=(mode != model_fn.ModeKeys.PREDICT))
      num_groups = array_ops.shape(mask)[1]

      with ops.name_scope('group_features'):
        # For context features, We have shape [batch_size * num_groups, ...].
        large_batch_context_features = {}
        for name, value in six.iteritems(context_features):
          # [batch_size, 1, ...].
          value = array_ops.expand_dims(value, axis=1)
          # [batch_size, num_groups, ...].
          value = array_ops.gather(
              value, array_ops.zeros([num_groups], dtypes.int32), axis=1)
          # [batch_size * num_groups, ...]
          large_batch_context_features[name] = utils.reshape_first_ndims(
              value, 2, [batch_size * num_groups])

        # For example feature, we have shape [batch_size * num_groups,
        # group_size, ...].
        large_batch_group_features = {}
        for name, value in six.iteritems(per_example_features):
          # [batch_size, num_groups, group_size, ...].
          value = array_ops.gather_nd(value, indices)
          # [batch_size * num_groups, group_size, ...].
          large_batch_group_features[name] = utils.reshape_first_ndims(
              value, 3, [batch_size * num_groups, group_size])

      # Do the inference and get scores for the large batch.
      # [batch_size * num_groups, group_size].
      scores = _score_fn(
          large_batch_context_features, large_batch_group_features, reuse=False)

      with ops.name_scope('accumulate_scores'):
        scores = array_ops.reshape(scores, [batch_size, num_groups, group_size])
        # Reset invalid scores to 0 based on mask.
        scores = array_ops.where(
            array_ops.gather(
                array_ops.expand_dims(mask, 2),
                array_ops.zeros([group_size], dtypes.int32),
                axis=2), scores, array_ops.zeros_like(scores))
        # [batch_size, num_groups, group_size].
        list_scores = array_ops.scatter_nd(indices, scores,
                                           [batch_size, list_size])
        # Use average.
        list_scores /= math_ops.to_float(group_size)

    if mode == model_fn.ModeKeys.PREDICT:
      return list_scores
    else:
      features.update(context_features)
      features.update(per_example_features)
      return list_scores

  def _model_fn(features, labels, mode, params, config):
    """Defines an `Estimator` model_fn."""
    params = params or {}

    tf_logging.info('Use groupwise dnn v2.')
    logits = _groupwise_dnn_v2(features, labels, mode, params, config)

    return ranking_head.create_estimator_spec(
        features=features, mode=mode, logits=logits, labels=labels)

  return _model_fn
