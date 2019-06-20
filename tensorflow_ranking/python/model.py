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

import abc
import six
import tensorflow as tf

from tensorflow.python.util import function_utils

from tensorflow_ranking.python import feature
from tensorflow_ranking.python import utils


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
        * Returns:
          `context_features`: A dict of `Tensor`s with shape [batch_size, ...]
          `example_features`: A dict of `Tensor`s with shape [batch_size,
            list_size, ...]
    """
    if transform_fn is None:
      self._transform_fn = feature.make_identity_transform_fn({})
    else:
      self._transform_fn = transform_fn

  def _call_transform_fn(self, features, mode):
    """Calls transform_fn and returns dense Tensors."""
    transform_fn_args = function_utils.fn_args(self._transform_fn)
    if 'mode' in transform_fn_args:
      return self._transform_fn(features, mode=mode)
    else:
      return self._transform_fn(features)

  def compute_logits(self, features, labels, mode, params, config):
    """Computes the logits for the given inputs.

    This is the core interface function for a ranking model and it computes the
    logits for a list/set of examples.  The logic of a specific ranking model
    should be implemented in `_compute_logits_impl`. For example, in the
    simplest case, the logits contain a float value for each example and we also
    have a single label for each example. In a more advanced multi-task setting,
    logits can contain a vector of floats for each example and so do the labels.

    Args:
      features: (dict) A dict of Tensors or SparseTensors of shape [batch_size,
        list_size, ...] for example features and shape [batch_size, ...] for
        context features.
      labels: (Tensor) A dense Tensor representing relevance for the ranking
        problem.
      mode: See tf.estimator.ModeKeys.
      params: See tf.estimator model_fn. Hyperparameters for the model.
      config: See tf.estimator model_fn.

    Returns:
      A Tensor representing the logits.
    """
    with tf.compat.v1.name_scope('transform'):
      context_features, example_features = self._call_transform_fn(
          features, mode)
      # Check feature tensor shape.
      for name, value in six.iteritems(example_features):
        tensor_shape = tf.convert_to_tensor(value).shape
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
    rw_size: A scalr int `Tensor` for the rw_size.
    num_valid_entries: A 1-D `Tensor` with shape [batch_size] representing the
      number of valid entries for each instance in a batch.

  Returns:
    A tuple of Tensors (batch_rw_indices, batch_indices_mask). The first has
    shape [batch_size, size, rw_size] and the second has shape [batch_size,
    size].
  """
  with tf.name_scope(name='rolling_window_indices'):
    # shape = [size, rw_size] with value [[0, 1, ...], [1, 2, ...], ...].
    rw_indices = tf.expand_dims(tf.range(rw_size), 0) + tf.expand_dims(
        tf.range(size), 1)
    # shape = [batch_size, size, rw_size]. Make batch_size copies.
    batch_rw_indices = tf.gather(
        tf.expand_dims(rw_indices, 0), tf.zeros_like(num_valid_entries), axis=0)
    # Mark the first n indices as valid where n = num_valid_entries.
    batch_indices_mask = tf.less(
        tf.reduce_min(input_tensor=batch_rw_indices, axis=2),
        tf.reshape(num_valid_entries, [-1, 1]))
    # Mod the indices to the range of num_valid_entries.
    num_valid_entries = tf.where(
        tf.less(num_valid_entries, 1), tf.ones_like(num_valid_entries),
        num_valid_entries)
    batch_rw_indices = tf.mod(batch_rw_indices,
                              tf.reshape(num_valid_entries, [-1, 1, 1]))
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
  with tf.name_scope(name='form_group_indices'):
    is_valid = tf.convert_to_tensor(value=is_valid)
    batch_size, list_size = tf.unstack(tf.shape(input=is_valid))
    num_valid_entries = tf.reduce_sum(
        input_tensor=tf.cast(is_valid, dtype=tf.int32), axis=1)
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
        * Returns: Tensor of shape [batch_size, group_size] that contains a
          score for each example.
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
    self._feature_gather_indices = None  # Internal.
    self._score_scatter_indices = None  # Internal.
    self._indices_mask = None  # Internal.

  def _update_scatter_gather_indices(self, is_valid, mode):
    """Updates the internal scatter/gather indices."""
    shuffle = not (self._group_size == 1 or
                   mode == tf.estimator.ModeKeys.PREDICT)
    (self._feature_gather_indices, self._indices_mask) = _form_group_indices_nd(
        is_valid, self._group_size, shuffle=shuffle)
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
      self._update_scatter_gather_indices(is_valid, mode)
      num_groups = tf.shape(input=self._indices_mask)[1]

      with tf.compat.v1.name_scope('group_features'):
        # For context features, We have shape [batch_size * num_groups, ...].
        large_batch_context_features = {}
        for name, value in six.iteritems(context_features):
          # [batch_size, 1, ...].
          value = tf.expand_dims(value, axis=1)
          # [batch_size, num_groups, ...].
          value = tf.gather(value, tf.zeros([num_groups], tf.int32), axis=1)
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
        scores = tf.reshape(scores, tf.shape(self._score_scatter_indices)[0:3])

      with tf.compat.v1.name_scope('accumulate_scores'):
        # Reset invalid scores to 0 based on mask.
        scores = tf.where(
            tf.gather(
                tf.expand_dims(self._indices_mask, 2),
                tf.zeros([tf.shape(scores)[2]], tf.int32),
                axis=2), scores, tf.zeros_like(scores))
        # Scatter scores from [batch_size, num_groups, logits_size] to
        # [batch_size, list_size].
        logits = tf.scatter_nd(self._score_scatter_indices, scores,
                               [batch_size, list_size])
        # Use average.
        logits /= tf.cast(tf.shape(scores)[2], dtype=tf.float32)
    return logits


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
    An `Estimator` `model_fn` with the following signature:
    * Args:
      `features`: The raw features from input_fn.
      `labels`: A Tensor with shape [batch_size, list_size].
      `mode`: No difference.
      `params`: No difference.
      `config`: No difference..
    * Returns:
      `EstimatorSpec`.
  """

  tf.compat.v1.logging.info('Building groupwise ranking model.')
  ranking_model = _GroupwiseRankingModel(group_score_fn, group_size,
                                         transform_fn)

  def _model_fn(features, labels, mode, params, config):
    """Defines an `Estimator` model_fn."""
    logits = ranking_model.compute_logits(features, labels, mode, params,
                                          config)
    return ranking_head.create_estimator_spec(
        features=features, mode=mode, logits=logits, labels=labels)

  return _model_fn
