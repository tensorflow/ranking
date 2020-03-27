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

"""Tensorflow Ranking model library.

This provides functions to build `model_fn` used in `tf.estimator`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

from tensorflow.python.util import function_utils

from tensorflow_ranking.python import feature
from tensorflow_ranking.python import utils

# Constant names in `params`.
# The following are parameter names for number of shuffles of the lists.
_NUM_SHUFFLES_TRAIN = 'num_shuffles_train'
_NUM_SHUFFLES_EVAL = 'num_shuffles_eval'
_NUM_SHUFFLES_PREDICT = 'num_shuffles_predict'


def _get_params(mode, params):
  """Returns the params defined by the above constants."""
  params = params or {}
  if mode == tf.estimator.ModeKeys.TRAIN:
    num_shuffles = params.get(_NUM_SHUFFLES_TRAIN, None)
  elif mode == tf.estimator.ModeKeys.EVAL:
    num_shuffles = params.get(_NUM_SHUFFLES_EVAL, None)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    num_shuffles = params.get(_NUM_SHUFFLES_PREDICT, None)
  else:
    raise ValueError('Invalid mode: {}.'.format(mode))
  return num_shuffles


class _RankingModel(object):
  """Interface for a ranking model."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, transform_fn=None):
    """Constructor for the common components of all ranking models.

    Args:
      transform_fn: (function) A user-provided function that transforms raw
        features into dense Tensors with the following signature:
        * Args:
          `features`: A dict of Tensors or SparseTensors that contains the raw
            features from an input_fn.
          `mode`: Optional. See estimator `ModeKeys`.
          `params`: Optional. See tf.estimator model_fn. Hyperparameters for the
            model.
        * Returns:
          `context_features`: A dict of `Tensor`s with shape [batch_size, ...]
          `example_features`: A dict of `Tensor`s with shape [batch_size,
            list_size, ...]
    """
    if transform_fn is None:
      self._transform_fn = feature.make_identity_transform_fn({})
    else:
      self._transform_fn = transform_fn

  def _call_transform_fn(self, features, mode, params):
    """Calls transform_fn and returns dense Tensors."""
    transform_fn_args = function_utils.fn_args(self._transform_fn)
    if 'mode' in transform_fn_args and 'params' in transform_fn_args:
      return self._transform_fn(features, mode=mode, params=params)
    elif 'mode' in transform_fn_args:
      return self._transform_fn(features, mode=mode)
    elif 'params' in transform_fn_args:
      return self._transform_fn(features, params=params)
    else:
      return self._transform_fn(features)

  def compute_logits(self, features, labels, mode, params, config):
    """Computes the logits for the given inputs.

    This is the core interface function for a ranking model and it computes the
    logits for a list/set of examples.  The logic of a specific ranking model
    should be implemented in `_compute_logits_impl`. For example, in the
    simplest case, the logits contain a float value for each example and we also
    have a single label for each example. In a more advanced multi-task setting,
    logits are a `dict` that maps task name to task-specific logits.

    Args:
      features: (dict) A dict of Tensors or SparseTensors of shape [batch_size,
        list_size, ...] for example features and shape [batch_size, ...] for
        context features.
      labels: (Tensor or dict) A dense Tensor representing relevance for the
        ranking problem or a `dict` of Tensors in the multi-task setting.
      mode: See tf.estimator.ModeKeys.
      params: See tf.estimator model_fn. Hyperparameters for the model.
      config: See tf.estimator model_fn.

    Returns:
      A dict of Tensors for multi-task or a Tensor for single task that
      represents the logits.
    """
    with tf.compat.v1.name_scope('transform'):
      context_features, example_features = self._call_transform_fn(
          features, mode, params)
      # Check feature tensor shape.
      for name, value in six.iteritems(example_features):
        tensor_shape = tf.convert_to_tensor(value=value).shape
        if (tensor_shape is not None and tensor_shape.rank is not None and
            tensor_shape.rank < 3):
          tf.compat.v1.logging.warning(
              'Feature \"{}\" has invalid feature tensor shape {}. '
              'Expected shape has at least 3 dims: '
              '(batch_size, list_size, feature_size).'.format(
                  name, tensor_shape))

    logits = self._compute_logits_impl(context_features, example_features,
                                       labels, mode, params, config)

    if mode == tf.estimator.ModeKeys.PREDICT:
      return logits
    else:
      features.update(context_features)
      features.update(example_features)
      return logits

  @abc.abstractmethod
  def _compute_logits_impl(self, context_features, example_features, labels,
                           mode, params, config):
    """Implements the logic that computes the logits from input dense Tensors.

    Args:
      context_features: (dict) A dict of dense Tensors for context features.
        Each Tensor is 2-D and has a shape of [batch_size, feature_size].
      example_features: (dict) A dict of dense Tensors for example features.
        Each Tensor id 3-D and has a shape of [batch_size, list_size,
        feature_size].
      labels: See `compute_logits`.
      mode: See `compute_logits`.
      params: See `compute_logits`.
      config: See `compute_logits`.

    Returns:
      A Tensor representing the logits.
    """
    raise NotImplementedError('Calling an abstract method.')


def _rolling_window_indices(size, rw_size, num_valid_entries):
  """Returns the rolling windows indices and mask for valid ones.

  When size = 3, rw_size = 2, returns [[0, 1], [1, 2], [2, 0]]. When size = 2,
  rw_size = 3, returns [[0, 1, 0], [1, 0, 1]].

  When num_valid_entries = 2, the first returns [[0, 1], [1, 0], [0, 1]] and the
  first 2 are valid with mask as [True, True, False].

  Args:
    size: A scalar int `Tensor` for the size.
    rw_size: A scalar int `Tensor` for the rw_size.
    num_valid_entries: A 1-D `Tensor` with shape [batch_size] representing the
      number of valid entries for each instance in a batch.

  Returns:
    A tuple of Tensors (batch_rw_indices, batch_indices_mask). The first has
    shape [batch_size, size, rw_size] and the second has shape [batch_size,
    size].
  """
  with tf.compat.v1.name_scope(name='rolling_window_indices'):
    # shape = [size, rw_size] with value [[0, 1, ...], [1, 2, ...], ...].
    rw_indices = tf.expand_dims(tf.range(rw_size), 0) + tf.expand_dims(
        tf.range(size), 1)
    # shape = [batch_size, size, rw_size]. Make batch_size copies.
    batch_size = tf.shape(input=num_valid_entries)[0]
    batch_rw_indices = tf.tile(
        tf.expand_dims(rw_indices, 0), multiples=[batch_size, 1, 1])
    # Mark the first n indices as valid where n = num_valid_entries.
    batch_indices_mask = tf.less(
        tf.reduce_min(input_tensor=batch_rw_indices, axis=2),
        tf.reshape(num_valid_entries, [-1, 1]))
    # Mod the indices to the range of num_valid_entries.
    num_valid_entries = tf.compat.v1.where(
        tf.less(num_valid_entries, 1), tf.ones_like(num_valid_entries),
        num_valid_entries)
    batch_rw_indices = tf.math.mod(batch_rw_indices,
                                   tf.reshape(num_valid_entries, [-1, 1, 1]))
    return batch_rw_indices, batch_indices_mask


def _form_group_indices_nd(is_valid, group_size, shuffle=False, seed=None):
  """Forms the indices for groups for gather_nd or scatter_nd.

  Args:
    is_valid: A boolean `Tensor` for entry validity with shape [batch_size,
      list_size].
    group_size: An scalar int `Tensor` for the number of examples in a group.
    shuffle: A boolean that indicates whether valid indices should be shuffled
      when forming group indices.
    seed: Random seed for shuffle.

  Returns:
    A tuple of Tensors (indices, mask). The first has shape [batch_size,
    num_groups, group_size, 2] and it can be used in gather_nd or scatter_nd for
    group features. The second has the shape of [batch_size, num_groups] with
    value True for valid groups.
  """
  with tf.compat.v1.name_scope(name='form_group_indices'):
    is_valid = tf.convert_to_tensor(value=is_valid)
    batch_size, list_size = tf.unstack(tf.shape(input=is_valid))
    num_valid_entries = tf.reduce_sum(
        input_tensor=tf.cast(is_valid, dtype=tf.int32), axis=1)
    rw_indices, mask = _rolling_window_indices(list_size, group_size,
                                               num_valid_entries)
    # Valid indices of the tensor are shuffled and put on the top.
    # [batch_size, list_size, 2]. A deterministic op-level seed is set mainly
    # for unittest purpose. We can find a better way to avoid setting this seed
    # explicitly.
    shuffled_indices = utils.organize_valid_indices(
        is_valid, shuffle=shuffle, seed=seed)
    # Construct indices for gather_nd.
    # [batch_size, num_groups, group_size, 2]
    group_indices_nd = tf.expand_dims(rw_indices, axis=3)
    group_indices_nd = tf.concat([
        tf.reshape(tf.range(batch_size), [-1, 1, 1, 1]) *
        tf.ones_like(group_indices_nd), group_indices_nd
    ], 3)

    indices = tf.gather_nd(shuffled_indices, group_indices_nd)
    return indices, mask


def _infer_sizes(example_features, labels):
  """Infers batch_size, list_size, and is_valid based on inputs."""
  with tf.compat.v1.name_scope('infer_sizes'):
    if labels is not None:
      if isinstance(labels, dict):
        labels = next(six.itervalues(labels))
      batch_size, list_size = tf.unstack(tf.shape(input=labels))
      is_valid = utils.is_label_valid(labels)
    else:
      if not example_features:
        raise ValueError('`example_features` is empty.')

      # Infer batch_size and list_size from a feature.
      example_tensor_shape = tf.shape(
          input=next(six.itervalues(example_features)))
      batch_size = example_tensor_shape[0]
      list_size = example_tensor_shape[1]
      # Mark all entries as valid in case we don't have enough information.
      # TODO: Be more smart to infer is_valid.
      is_valid = utils.is_label_valid(tf.ones([batch_size, list_size]))
  if batch_size is None or list_size is None:
    raise ValueError('Invalid batch_size=%s or list_size=%s' %
                     (batch_size, list_size))
  return batch_size, list_size, is_valid


class _GroupwiseRankingModel(_RankingModel):
  """Ranking model for groupwise scoring functions."""

  def __init__(self, group_score_fn, group_size, transform_fn=None):
    """Constructor for groupwise ranking model.

    Args:
      group_score_fn: A scoring function for a `group_size` number of examples
        with the following signature:
        * Args:
          `context_features`: A dict of `Tensor`s with shape [batch_size, ...].
          `group_features`: A dict of `Tensor`s with shape [batch_size,
            group_size, ...]
          `mode`: Optional. Specifies if this is training, evaluation or
            inference. See `ModeKeys`.
          `params`: Optional dict of hyperparameters, same value passed in the
            `Estimator` constructor.
          `config`: Optional configuration object, same value passed in the
            `Estimator` constructor.
        * Returns: A Tensor of shape [batch_size, group_size] that contains a
          score for each example, or a `dict` of Tensors with the above shape in
          multi-task setting.
      group_size: An integer denoting the number of examples in
        `group_score_fn`.
      transform_fn: See `_RankingModel`.

    Raises:
      ValueError: when group_size is invalid.
    """
    super(_GroupwiseRankingModel, self).__init__(transform_fn)
    if group_size <= 0:
      raise ValueError('Invalid group_size %d' % group_size)
    self._group_size = group_size
    self._score_fn = group_score_fn

    # Internal variables.
    self._feature_gather_indices = None
    self._score_scatter_indices = None
    self._indices_mask = None

  def _update_scatter_gather_indices(self, is_valid, mode, params):
    """Updates the internal scatter/gather indices."""
    num_shuffles = _get_params(mode, params)
    if self._group_size == 1:
      shuffle = False
      num_shuffles = None
    elif mode == tf.estimator.ModeKeys.PREDICT:
      shuffle = num_shuffles is not None
    else:
      shuffle = True

    # Shuffle the indices the `num_shuffles` times and concat shuffled indices.
    num_shuffles = num_shuffles or 1
    assert num_shuffles > 0, 'Invalid num_shuffles: {}'.format(num_shuffles)
    if shuffle:
      tf.compat.v1.logging.info('Number of shuffles: {}'.format(num_shuffles))
    indices_shuffled = []
    # Use fixed ops-level seeds so that the randomness is controlled by the
    # graph-level seed.
    for i in range(num_shuffles):
      indices_shuffled.append(
          _form_group_indices_nd(
              is_valid, self._group_size, shuffle=shuffle, seed=i + 77))
    feature_gather_indices_list, indices_mask_list = zip(*indices_shuffled)
    self._feature_gather_indices = tf.concat(feature_gather_indices_list, 1)
    self._indices_mask = tf.concat(indices_mask_list, 1)
    self._score_scatter_indices = self._feature_gather_indices

  def _compute_logits_impl(self, context_features, example_features, labels,
                           mode, params, config):
    # Scatter/Gather per-example scores through groupwise comparison. Each
    # instance in a mini-batch will form a number of groups. Each group of
    # examples are scored by `_score_fn` and scores for individual examples are
    # accumulated into logits.
    with tf.compat.v1.name_scope('groupwise_dnn_v2'):
      batch_size, list_size, is_valid = _infer_sizes(example_features, labels)
      # For each example feature, assuming the shape is [batch_size, list_size,
      # feature_size], the groups are formed along the 2nd dim. Each group has a
      # 'group_size' number of indices in [0, list_size). Based on these
      # indices, we can gather the example feature into a sub-tensor for each
      # group. The total number of groups we have for a mini-batch is batch_size
      # * num_groups. Inside each group, we have a 'group_size' number of
      # examples.
      self._update_scatter_gather_indices(is_valid, mode, params)
      num_groups = tf.shape(input=self._indices_mask)[1]

      with tf.compat.v1.name_scope('group_features'):
        # For context features, We have shape [batch_size * num_groups, ...].
        large_batch_context_features = {}
        for name, value in six.iteritems(context_features):
          # [batch_size, num_groups, ...].
          value = tf.repeat(
              tf.expand_dims(value, axis=1), repeats=[num_groups], axis=1)
          # [batch_size * num_groups, ...]
          large_batch_context_features[name] = utils.reshape_first_ndims(
              value, 2, [batch_size * num_groups])

        # For example feature, we have shape [batch_size * num_groups,
        # group_size, ...].
        large_batch_group_features = {}
        for name, value in six.iteritems(example_features):
          # [batch_size, num_groups, group_size, ...].
          value = tf.gather_nd(value, self._feature_gather_indices)
          # [batch_size * num_groups, group_size, ...].
          large_batch_group_features[name] = utils.reshape_first_ndims(
              value, 3, [batch_size * num_groups, self._group_size])

      # Do the inference and get scores for the large batch of [batch_size *
      # num_groups, logits_size] and reshape them to [batch_size, num_groups,
      # logits_size].
      with tf.compat.v1.variable_scope('group_score'):
        scores = self._score_fn(large_batch_context_features,
                                large_batch_group_features, mode, params,
                                config)

      with tf.compat.v1.name_scope('accumulate_scores'):
        # Reset invalid scores to 0 based on mask.
        scores_mask = tf.tile(
            tf.expand_dims(self._indices_mask, 2),
            multiples=[1, 1,
                       tf.shape(input=self._score_scatter_indices)[2]],
            name='tile_scores_mask')
        counts = tf.scatter_nd(self._score_scatter_indices,
                               tf.cast(scores_mask, tf.float32),
                               [batch_size, list_size])

        def _accumulate_scores(task_scores):
          """A subroutine to accumulate scores for a single Tensor."""
          task_scores = tf.reshape(
              task_scores,
              tf.shape(input=self._score_scatter_indices)[0:3])
          task_scores = tf.compat.v1.where(scores_mask, task_scores,
                                           tf.zeros_like(task_scores))
          # Scatter scores from [batch_size, num_groups, group_size] to
          # [batch_size, list_size].
          task_logits = tf.scatter_nd(self._score_scatter_indices, task_scores,
                                      [batch_size, list_size])
          # Use average.
          task_logits = tf.compat.v1.div_no_nan(task_logits, counts)
          return task_logits

        if isinstance(scores, dict):
          logits = {}
          for name, task_scores in six.iteritems(scores):
            logits[name] = _accumulate_scores(task_scores)
        else:
          logits = _accumulate_scores(scores)

    return logits


def _make_model_fn(ranking_model, ranking_head):
  """A helper function to make an `Estimator` model_fn.

  Args:
    ranking_model: A `_RankingModel` object.
    ranking_head: A `head._RankingHead` object.

  Returns:
    An `Estimator` `model_fn` with the following signature:
    * Args:
      `features`: The raw features from input_fn.
      `labels`: A Tensor with shape [batch_size, list_size] or a dict of Tensors
        in multi-task setting.
      `mode`: No difference.
      `params`: No difference.
      `config`: No difference..
    * Returns:
      `EstimatorSpec`.
  """

  def _model_fn(features, labels, mode, params, config):
    """Defines an `Estimator` `model_fn`."""
    logits = ranking_model.compute_logits(features, labels, mode, params,
                                          config)
    return ranking_head.create_estimator_spec(
        features=features, mode=mode, logits=logits, labels=labels)

  return _model_fn


def make_groupwise_ranking_fn(group_score_fn,
                              group_size,
                              ranking_head,
                              transform_fn=None):
  """Builds an `Estimator` model_fn for groupwise comparison ranking models.

  Args:
    group_score_fn: See `_GroupwiseRankingModel`.
    group_size: See `_GroupwiseRankingModel`.
    ranking_head: A `head._RankingHead` object.
    transform_fn: See `_GroupwiseRankingModel`.

  Returns:
    See `_make_model_fn`.
  """

  tf.compat.v1.logging.info('Building groupwise ranking model.')
  ranking_model = _GroupwiseRankingModel(group_score_fn, group_size,
                                         transform_fn)
  return _make_model_fn(ranking_model, ranking_head)
