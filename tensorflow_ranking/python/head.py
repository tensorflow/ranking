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

"""Defines `Head`s of TF ranking models.

Given logits (or output of a hidden layer), a `Head` computes predictions,
loss, train_op, metrics and exports outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

_DEFAULT_SERVING_KEY = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# The above default is defined by TF Serving, but these next two are just a
# local convention without any special meaning.
_REGRESS_SERVING_KEY = 'regression'
_PREDICT_SERVING_KEY = 'predict'


def create_ranking_head(loss_fn,
                        eval_metric_fns=None,
                        optimizer=None,
                        train_op_fn=None,
                        name=None):
  """A factory method to create `_RankingHead`.

  Args:
    loss_fn: A loss function with the following signature (see make_loss_fn in
      losses.py):
      * Args:
        `labels`: A `Tensor` of the same shape as `logits` representing
          relevance.
        `logits`: A `Tensor` with shape [batch_size, list_size]. Each value is
          the ranking score of the corresponding example.
        `features`: A dict of `Tensor`s for all features.
      * Returns: A scalar containing the loss to be optimized.
    eval_metric_fns: A dict of metric functions keyed by a string name. The
      values of the dict are metric functions with the following signature:
      * Args:
        `labels`: A `Tensor` of the same shape as `predictions` representing
          relevance.
        `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
          is the ranking score of the corresponding example.
        `features`: A dict of `Tensor`s for all features.
      * Returns: The result of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple.
    optimizer: `Optimizer` instance used to optimize the loss in TRAIN mode.
      Namely, it sets `train_op = optimizer.minimize(loss, global_step)`, which
      updates variables and increments `global_step`.
    train_op_fn: Function that takes a scalar loss `Tensor` and returns
      `train_op`. Used if `optimizer` is `None`.
    name: Name of the head. If provided, will be used as `name_scope` when
      creating ops.

  Returns:
    An instance of `_RankingHead` for ranking.

  Raises:
    ValueError: If `loss_fn` is not callable.
  """
  if not callable(loss_fn):
    raise ValueError('Not callable loss_fn: {}'.format(loss_fn))

  return _RankingHead(
      loss_fn=loss_fn,
      eval_metric_fns=eval_metric_fns,
      optimizer=optimizer,
      train_op_fn=train_op_fn,
      name=name)


def create_multi_ranking_head(heads, head_weights=None):
  """A factory method to create `_MultiRankingHead`.

  Args:
    heads: A tuple or list of `_RankingHead`.
    head_weights: A tuple or list of weights.

  Returns:
    An instance of `_MultiRankingHead` for multi-task learning.
  """
  return _MultiRankingHead(heads, head_weights)


class _AbstractRankingHead(object):
  """Interface for ranking head."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The head name."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: A `Tensor` or a dict of (name, `Tensor`). Each `Tensor` has the
        shape of [batch_size, D]. Each value is the ranking score of the
        corresponding item. `D` is usually the `list_size`. It might be changed
        when `mode` is `PREDICT`. When `logits` is a dict, it is for multi-task
        setting.
      labels: A `Tensor` or a dict of (name, `Tensor`) representing relevance
        labels. Each `Tensor` has the same shape as `logits`. `labels` is
        required argument when `mode` equals `TRAIN` or `EVAL`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.

    Returns:
      `EstimatorSpec`.
    """
    raise NotImplementedError('Calling an abstract method.')


def _labels_and_logits_metrics(labels, logits):
  """Returns metrics for labels and logits."""
  is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1])
  metrics_dict = {}
  for name, tensor in [('labels_mean', labels), ('logits_mean', logits)]:
    metrics_dict[name] = tf.compat.v1.metrics.mean(
        tf.boolean_mask(tensor=tf.reshape(tensor, [-1]), mask=is_label_valid))
  return metrics_dict


def _get_train_op(loss, train_op_fn=None, optimizer=None):
  """Returns a train op."""
  if optimizer is not None:
    if train_op_fn is not None:
      raise ValueError('train_op_fn and optimizer cannot both be set.')
    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_global_step())
  elif train_op_fn is not None:
    train_op = train_op_fn(loss)
  else:
    raise ValueError('train_op_fn and optimizer cannot both be None.')
  return train_op


class _RankingHead(_AbstractRankingHead):
  """Interface for the head/top of a ranking model."""

  def __init__(self,
               loss_fn,
               eval_metric_fns=None,
               optimizer=None,
               train_op_fn=None,
               name=None):
    """Constructor. See `create_ranking_head`."""
    self._loss_fn = loss_fn
    self._eval_metric_fns = eval_metric_fns or {}
    self._optimizer = optimizer
    self._train_op_fn = train_op_fn
    self._name = name

  @property
  def name(self):
    return self._name

  def create_loss(self, features, mode, logits, labels):
    """Returns a loss Tensor from provided logits.

    This function is designed to be used by framework developers.  Almost all
    users should use create_estimator_spec(), which calls this internally.
    `mode` and `features` are most likely not used, but some Head
    implementations may require them.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used for loss construction.
      labels: Labels `Tensor`, or `dict` of same.

    Returns:
      A LossSpec object.
    """
    del mode  # Unused for this head.
    logits = tf.convert_to_tensor(value=logits)
    labels = tf.cast(labels, dtype=tf.float32)

    training_loss = self._loss_fn(labels, logits, features)

    return training_loss

  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            regularization_losses=None):
    """See `_AbstractRankingHead`."""
    logits = tf.convert_to_tensor(value=logits)
    # Predict.
    with tf.compat.v1.name_scope(self._name, 'head'):
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits,
            export_outputs={
                _DEFAULT_SERVING_KEY:
                    tf.estimator.export.RegressionOutput(logits),
                _REGRESS_SERVING_KEY:
                    tf.estimator.export.RegressionOutput(logits),
                _PREDICT_SERVING_KEY:
                    tf.estimator.export.PredictOutput(logits),
            })

      training_loss = self.create_loss(
          features=features, mode=mode, logits=logits, labels=labels)
      if regularization_losses:
        regularization_loss = tf.add_n(regularization_losses)
        regularized_training_loss = tf.add(training_loss, regularization_loss)
      else:
        regularized_training_loss = training_loss

      # Eval.
      if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            name:
            metric_fn(labels=labels, predictions=logits, features=features)
            for name, metric_fn in six.iteritems(self._eval_metric_fns)
        }
        eval_metric_ops.update(_labels_and_logits_metrics(labels, logits))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits,
            loss=regularized_training_loss,
            eval_metric_ops=eval_metric_ops)

      # Train.
      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=regularized_training_loss,
            train_op=_get_train_op(regularized_training_loss, self._train_op_fn,
                                   self._optimizer),
            predictions=logits)
      raise ValueError('mode={} unrecognized'.format(mode))


def _default_export_output(export_outputs, head_name):
  """Extracts the default export output from the given export_outputs dict."""
  if len(export_outputs) == 1:
    return next(six.itervalues(export_outputs))
  try:
    return export_outputs[_DEFAULT_SERVING_KEY]
  except KeyError:
    raise ValueError(
        '{} did not specify default export_outputs. '
        'Given: {} '
        'Suggested fix: Use one of the heads in tf.estimator, or include '
        'key {} in export_outputs.'.format(head_name, export_outputs,
                                           _DEFAULT_SERVING_KEY))


class _MultiRankingHead(_AbstractRankingHead):
  """Composition of multiple `_RankingHead.

  The main functionality is to create an estimator spec that
    - merges all predictions from all heads.
    - merges all eval metrics from all heads.
    - combines all the losses using weights from all heads.
  """

  def __init__(self, heads, head_weights=None):
    """Constructor.

    Args:
      heads: A tuple or list of `_RankingHead`.
      head_weights: A tuple or list of weights.
    """
    if not heads:
      raise ValueError('Must specify heads. Given: {}'.format(heads))
    if head_weights:
      if len(head_weights) != len(heads):
        raise ValueError(
            'heads and head_weights must have the same size. '
            'Given len(heads): {}. Given len(head_weights): {}.'.format(
                len(heads), len(head_weights)))
    for head in heads:
      if head.name is None:
        raise ValueError(
            'All given heads must have name specified. Given: {}'.format(head))
    self._heads = tuple(heads)
    self._head_weights = tuple(head_weights) if head_weights else tuple()
    # TODO: Figure out a better way to set train_op_fn and optimizer
    # for _MultiRankingHead.
    # pylint: disable=protected-access
    tf.compat.v1.logging.info(
        'Use the train_op_fn and optimizer from the first head.')
    self._train_op_fn = self._heads[0]._train_op_fn
    self._optimizer = self._heads[0]._optimizer
    # pylint: enable=protected-access

  @property
  def name(self):
    """See `_AbstractRankingHead`."""
    return '_'.join([h.name for h in self._heads])

  def _check_logits_and_labels(self, logits, labels=None):
    """Validates the keys of logits and labels."""
    head_names = []
    for head in self._heads:
      head_names.append(head.name)

    if len(head_names) != len(set(head_names)):
      raise ValueError('Duplicated names in heads.')

    # Check the logits keys.
    if not isinstance(logits, dict):
      raise ValueError('logits in _MultiRankingHead should be a dict.')
    logits_missing_names = list(set(head_names) - set(list(logits)))
    if logits_missing_names:
      raise ValueError('logits has missing values for head(s): {}.'.format(
          logits_missing_names))

    # Check the labels keys.
    if labels is not None:
      if not isinstance(labels, dict):
        raise ValueError('labels in _MultiRankingHead should be a dict.')
      labels_missing_names = list(set(head_names) - set(list(labels)))
      if labels_missing_names:
        raise ValueError('labels has missing values for head(s): {}.'.format(
            labels_missing_names))

  def _merge_predict_export_outputs(self, all_estimator_spec):
    """Merges list of `EstimatorSpec` export_outputs for PREDICT.

    For each individual head, its _DEFAULT_SERVING_KEY and _PREDICT_SERVING_KEY
    are extracted and merged for `export_outputs` in PREDICT mode of
    `EstimatorSpec`. By default, the first head is served.

    Args:
      all_estimator_spec: list of `EstimatorSpec` for the individual heads.

    Returns:
      A dict of merged export_outputs from all heads for PREDICT.
    """
    # The first head is used for serving by default.
    export_outputs = {
        _DEFAULT_SERVING_KEY:
            _default_export_output(all_estimator_spec[0].export_outputs,
                                   self._heads[0].name),
    }
    merged_predict_outputs = {}
    for head, spec in zip(self._heads, all_estimator_spec):
      for k, v in six.iteritems(spec.export_outputs):
        # Collect default serving key for export_outputs
        key = (
            head.name if k == _DEFAULT_SERVING_KEY else '{}/{}'.format(
                head.name, k))
        export_outputs[key] = v
        # Collect predict serving key for merged_predict_outputs
        if (k == _PREDICT_SERVING_KEY and
            isinstance(v, tf.estimator.export.PredictOutput)):
          for kp, vp in six.iteritems(v.outputs):
            merged_predict_outputs['{}/{}'.format(head.name, kp)] = vp
    export_outputs[_PREDICT_SERVING_KEY] = (
        tf.estimator.export.PredictOutput(merged_predict_outputs))
    return export_outputs

  def _merge_loss(self,
                  labels,
                  logits,
                  features=None,
                  mode=None,
                  regularization_losses=None):
    """Returns regularized training loss."""
    self._check_logits_and_labels(logits, labels)
    training_losses = []
    for head in self._heads:
      training_loss = head.create_loss(
          logits=logits[head.name],
          labels=labels[head.name],
          features=features,
          mode=mode)
      training_losses.append(training_loss)
    training_losses = tuple(training_losses)

    with tf.compat.v1.name_scope(
        'merge_losses',
        values=training_losses + (self._head_weights or tuple())):
      if self._head_weights:
        head_weighted_training_losses = []
        for training_loss, head_weight in zip(training_losses,
                                              self._head_weights):
          head_weighted_training_losses.append(
              tf.math.multiply(training_loss, head_weight))
        training_losses = head_weighted_training_losses
      merged_training_loss = tf.math.add_n(training_losses)
      regularization_loss = tf.math.add_n(
          regularization_losses) if regularization_losses is not None else None
      regularized_training_loss = (
          merged_training_loss + regularization_loss
          if regularization_loss is not None else merged_training_loss)
    return regularized_training_loss

  def _merge_metrics(self, all_estimator_spec):
    """Merges the eval metrics from all heads."""
    # TODO: Add the per-head loss loss/head_name to metrics.
    eval_metric_ops = {}
    for head, spec in zip(self._heads, all_estimator_spec):
      eval_metric_ops.update({
          '{}/{}'.format(head.name, name): op
          for name, op in six.iteritems(spec.eval_metric_ops)
      })
    return eval_metric_ops

  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            regularization_losses=None):
    """See `_AbstractRankingHead`."""
    with tf.compat.v1.name_scope(self.name, 'multi_head'):
      self._check_logits_and_labels(logits, labels)
      # Get all estimator spec.
      all_estimator_spec = []
      for head in self._heads:
        all_estimator_spec.append(
            head.create_estimator_spec(
                features=features,
                mode=mode,
                logits=logits[head.name],
                labels=labels[head.name] if labels else None))
      # Predict.
      if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = self._merge_predict_export_outputs(all_estimator_spec)
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=logits, export_outputs=export_outputs)

      # Compute the merged loss and eval metrics.
      loss = self._merge_loss(labels, logits, features, mode,
                              regularization_losses)
      eval_metric_ops = self._merge_metrics(all_estimator_spec)

      # Eval.
      if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits,
            loss=loss,
            eval_metric_ops=eval_metric_ops)
      # Train.
      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=_get_train_op(loss, self._train_op_fn, self._optimizer),
            predictions=logits,
            eval_metric_ops=eval_metric_ops)
      raise ValueError('mode={} unrecognized'.format(mode))
