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

"""Defines `Head`s of TF ranking models.

Given logits (or output of a hidden layer), a `Head` computes predictions,
loss, train_op, metrics and exports outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_core
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import training_util
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.export import export_lib

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# The above default is defined by TF Serving, but these next three are just
# a local convention without any special meaning.
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


class _RankingHead(object):
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

  def _labels_and_logits_metrics(self, labels, logits):
    """Returns metrics for labels and logits."""
    is_label_valid = array_ops.reshape(math_ops.greater_equal(labels, 0.), [-1])
    return {
        name: metrics_core.mean(
            array_ops.boolean_mask(
                array_ops.reshape(tensor, [-1]), is_label_valid))
        for name, tensor in [('labels_mean', labels), ('logits_mean', logits)]
    }

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
      A LossSpec that contains
      * the scalar `Tensor` representing reduced weighted training loss
      * the `Tensor` representing the unreduced unweighted loss
      * the `Tensor` representing the example weights
      * possibly processed labels (e.g. vocabulary lookup, shape manipulation,
        etc.)
    """
    del mode  # Unused for this head.
    logits = ops.convert_to_tensor(logits)
    labels = math_ops.to_float(labels)

    training_loss = self._loss_fn(labels, logits, features)

    return head_lib.LossSpec(
        training_loss=training_loss,
        unreduced_loss=None,
        weights=None,
        processed_labels=labels)

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
      logits: A `Tensor` with shape [batch_size, D]. Each value is the ranking
        score of the corresponding item. `D` is usually the `list_size`. It
        might be changed when `mode` is `PREDICT`.
      labels: A `Tensor` of the same shape as `logits` representing relevance.
        `labels` is required argument when `mode` equals `TRAIN` or `EVAL`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.

    Returns:
      `EstimatorSpec`.
    Raises:
      ValueError: If, in TRAIN mode, both `train_op_fn` and `optimizer`
        specified in the init function are `None` or if both are set.
    """
    logits = ops.convert_to_tensor(logits)
    # Predict.
    with ops.name_scope(self._name, 'head'):
      if mode == model_fn.ModeKeys.PREDICT:
        return model_fn.EstimatorSpec(
            mode=mode,
            predictions=logits,
            export_outputs={
                _DEFAULT_SERVING_KEY: export_lib.RegressionOutput(logits),
                _REGRESS_SERVING_KEY: export_lib.RegressionOutput(logits),
                _PREDICT_SERVING_KEY: export_lib.PredictOutput(logits),
            })

      training_loss, _, _, _ = self.create_loss(
          features=features, mode=mode, logits=logits, labels=labels)
      if regularization_losses:
        regularization_loss = math_ops.add_n(regularization_losses)
        regularized_training_loss = math_ops.add(training_loss,
                                                 regularization_loss)
      else:
        regularized_training_loss = training_loss

      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        eval_metric_ops = {
            name: metric_fn(
                labels=labels, predictions=logits, features=features)
            for name, metric_fn in six.iteritems(self._eval_metric_fns)
        }
        eval_metric_ops.update(self._labels_and_logits_metrics(labels, logits))
        return model_fn.EstimatorSpec(
            mode=mode,
            predictions=logits,
            loss=regularized_training_loss,
            eval_metric_ops=eval_metric_ops)

      # Train.
      assert mode == model_fn.ModeKeys.TRAIN
      if self._optimizer is not None:
        if self._train_op_fn is not None:
          raise ValueError('train_op_fn and optimizer cannot both be set.')
        train_op = self._optimizer.minimize(
            regularized_training_loss,
            global_step=training_util.get_global_step())
      elif self._train_op_fn is not None:
        train_op = self._train_op_fn(regularized_training_loss)
      else:
        raise ValueError('train_op_fn and optimizer cannot both be None.')
      return model_fn.EstimatorSpec(
          mode=mode,
          predictions=logits,
          loss=regularized_training_loss,
          train_op=train_op)
