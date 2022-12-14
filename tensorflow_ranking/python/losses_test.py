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

"""Tests for ranking losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorflow_ranking.python import losses as ranking_losses
from tensorflow_ranking.python import losses_impl


def ln(x):
  return math.log(x)


def _pairwise_loss(labels, scores, weights, loss_form, rank_discount_form=None):
  """Returns the pairwise loss given the loss form.

  Args:
    labels: A list of graded relevance.
    scores: A list of item ranking scores.
    weights: A list of item weights.
    loss_form: A string representing the form of loss.
    rank_discount_form: A string representing the form of rank discount.

  Returns:
    A tuple of (sum of loss, sum of weights, count of nonzero weights).
  """
  scores, labels, weights = zip(
      *sorted(zip(scores, labels, weights), reverse=True))

  def _rank_discount(rank_discount_form, rank):
    discount = {
        'LINEAR': 1. / rank,
        'LOG': 1. / math.log(1. + rank),
    }
    return discount[rank_discount_form]

  def _lambda_weight(label_diff, i, j):
    delta = math.fabs(label_diff)
    if rank_discount_form is not None:
      delta *= math.fabs(
          _rank_discount(rank_discount_form, i + 1) -
          _rank_discount(rank_discount_form, j + 1))
    else:
      delta = 1. if delta > 0 else 0
    return delta

  def _loss(score_diff, label_diff, delta):
    if label_diff <= 0:
      return 0.
    loss_table = {
        ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS:
            max(0, 1 - score_diff) * delta,
        ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
            math.log(1. + math.exp(-score_diff)) * delta,
        ranking_losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
            1. / (1. + math.exp(score_diff)) * delta,
    }
    return loss_table[loss_form]

  loss = 0.
  weight = 0.
  count = 0.
  for i in range(len(labels)):
    for j in range(len(labels)):
      if labels[i] > labels[j]:
        delta = _lambda_weight(labels[i] - labels[j], i, j)
        part_loss = _loss(scores[i] - scores[j], labels[i] - labels[j], delta)
        if weights[i] > 0:
          loss += part_loss * weights[i]
          weight += delta * weights[i]
          if weight > 0:
            count += 1.
  return loss, weight, count


def _circle_loss(labels,
                 scores,
                 rank_discount_form=None,
                 gamma=64.,
                 margin=0.25):
  """Returns the circle loss given the loss form.

  Args:
    labels: A list of graded relevance.
    scores: A list of item ranking scores.
    rank_discount_form: A string representing the form of rank discount.
    gamma: A float parameter used in circle loss.
    margin: A float parameter defining the margin in circle loss.

  Returns:
    A tuple of (sum of loss, sum of lambda weights, count of nonzero weights).
  """
  scores, labels = zip(*sorted(zip(scores, labels), reverse=True))

  def _rank_discount(rank_discount_form, rank):
    discount = {
        'LINEAR': 1. / rank,
        'LOG': 1. / math.log(1. + rank),
    }
    return discount[rank_discount_form]

  def _lambda_weight(label_diff, i, j):
    delta = math.fabs(label_diff)
    if rank_discount_form is not None:
      delta *= math.fabs(
          _rank_discount(rank_discount_form, i + 1) -
          _rank_discount(rank_discount_form, j + 1))
    else:
      delta = 1. if delta > 0 else 0
    return delta

  def _loss(si, sj, label_diff, delta):
    if label_diff <= 0:
      return 0.
    return math.exp(gamma * max(0., (1 + margin) - si) *
                    ((1 - margin) - si) + gamma * max(0., sj + margin) *
                    (sj - margin)) * delta

  loss = 0.
  weight = 0.
  count = 0.
  for i in range(len(labels)):
    for j in range(len(labels)):
      if labels[i] > labels[j]:
        delta = _lambda_weight(labels[i] - labels[j], i, j)
        part_loss = _loss(scores[i], scores[j], labels[i] - labels[j], delta)
        loss += part_loss
        weight += delta
        if weight > 0:
          count += 1.
  return loss, weight, count


def _batch_aggregation(batch_loss_list, reduction=None):
  """Returns the aggregated loss."""
  loss_sum = 0.
  weight_sum = 0.
  for loss, weight, count in batch_loss_list:
    loss_sum += loss
    if reduction == 'mean':
      weight_sum += weight
    else:
      weight_sum += count
  return loss_sum / weight_sum


def _softmax(values):
  """Returns the softmax of `values`."""
  total = sum(math.exp(v) for v in values)
  return [math.exp(v) / (1e-20 + total) for v in values]


# Based on nn.sigmoid_cross_entropy_with_logits for x=logit and z=label the
# cross entropy is max(x, 0) - x * z + log(1 + exp(-abs(x)))
def _sigmoid_cross_entropy(labels, logits):

  def per_position_loss(logit, label):
    return max(logit, 0) - logit * label + math.log(1 + math.exp(-abs(logit)))

  return sum(
      per_position_loss(logit, label) for label, logit in zip(labels, logits))


def _neural_sort(logits, temperature=1.0):
  """Non-tensor version of neural sort."""
  batch_size = len(logits)
  list_size = len(logits[0])

  result = []
  for b in range(batch_size):
    smooth_perm = []
    logit_diff_sum = [sum(abs(m - l) for m in logits[b]) for l in logits[b]]
    for i in range(list_size):
      scaling = list_size + 1 - 2 * (i + 1)
      scaled_logits = [scaling * l for l in logits[b]]
      p_logits = [
          (l - s) / temperature for l, s in zip(scaled_logits, logit_diff_sum)
      ]
      p_logits = [l - max(p_logits) for l in p_logits]
      smooth_perm.append(_softmax(p_logits))
    result.append(smooth_perm)

  return result


def _softmax_cross_entropy(p_trues, p_preds):

  def per_list_loss(y_true, y_pred):
    return sum(-y_t * math.log(1e-20 + y_p) for y_t, y_p in zip(y_true, y_pred))

  return sum(
      per_list_loss(p_true, p_pred) for p_true, p_pred in zip(p_trues, p_preds))


# Aggregates the per position squared error.
def _mean_squared_error(logits, labels):
  return sum((logit - label)**2 for label, logit in zip(labels, logits))


class LossesTest(tf.test.TestCase):

  def _check_pairwise_loss(self, loss_fn):
    """Helper function to test `loss_fn`."""
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    listwise_weights = [[2.], [1.]]
    listwise_weights_expanded = [[2.] * 3, [1.] * 3]
    itemwise_weights = [[2., 3., 4.], [1., 1., 1.]]
    default_weights = [1.] * 3
    list_size = 3.
    loss_form_dict = {
        ranking_losses._pairwise_hinge_loss:
            ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
        ranking_losses._pairwise_logistic_loss:
            ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
        ranking_losses._pairwise_soft_zero_one_loss:
            ranking_losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS,
    }
    loss_form = loss_form_dict[loss_fn]
    # Individual lists.
    self.assertAlmostEqual(
        loss_fn([labels[0]], [scores[0]]).numpy(),
        _batch_aggregation(
            [_pairwise_loss(labels[0], scores[0], default_weights, loss_form)]),
        places=5)
    self.assertAlmostEqual(
        loss_fn([labels[1]], [scores[1]]).numpy(),
        _batch_aggregation(
            [_pairwise_loss(labels[1], scores[1], default_weights, loss_form)]),
        places=5)

    # Itemwise weights.
    self.assertAlmostEqual(
        loss_fn([labels[0]], [scores[0]],
                weights=[itemwise_weights[0]]).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], itemwise_weights[0], loss_form)
        ]),
        places=5)

    self.assertAlmostEqual(
        loss_fn([labels[1]], [scores[1]],
                weights=[itemwise_weights[1]]).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], itemwise_weights[1], loss_form)
        ]),
        places=5)

    # Multiple lists.
    self.assertAlmostEqual(
        loss_fn(labels, scores, weights=listwise_weights).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], listwise_weights_expanded[0],
                           loss_form),
            _pairwise_loss(labels[1], scores[1], listwise_weights_expanded[1],
                           loss_form)
        ]),
        places=5)

    # Test LambdaWeight.
    lambda_weight = losses_impl.DCGLambdaWeight(
        rank_discount_fn=lambda r: 1. / tf.math.log1p(r), smooth_fraction=1.)
    self.assertAlmostEqual(
        loss_fn(
            labels,
            scores,
            weights=listwise_weights,
            lambda_weight=lambda_weight).numpy(),
        _batch_aggregation([
            _pairwise_loss(
                labels[0],
                scores[0],
                listwise_weights_expanded[0],
                loss_form,
                rank_discount_form='LOG'),
            _pairwise_loss(
                labels[1],
                scores[1],
                listwise_weights_expanded[1],
                loss_form,
                rank_discount_form='LOG')
        ]) * list_size,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    reduced_1 = loss_fn(
        labels, scores, reduction=tf.compat.v1.losses.Reduction.SUM).numpy()
    reduced_2 = loss_fn(
        labels, scores, reduction=tf.compat.v1.losses.Reduction.MEAN).numpy()
    self.assertNotAlmostEqual(reduced_1, reduced_2)

  def _check_make_pairwise_loss(self, loss_key):
    """Helper function to test `make_loss_fn`."""
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    listwise_weights = [[2.], [1.]]
    listwise_weights_expanded = [[2.] * 3, [1.] * 3]
    itemwise_weights = [[2., 3., 4.], [1., 1., 1.]]
    default_weights = [1.] * 3
    weights_feature_name = 'weights'
    list_size = 3.
    features = {}

    loss_fn = ranking_losses.make_loss_fn(loss_key)
    # Individual lists.
    self.assertAlmostEqual(
        loss_fn([labels[0]], [scores[0]], features).numpy(),
        _batch_aggregation(
            [_pairwise_loss(labels[0], scores[0], default_weights, loss_key)]),
        places=5)
    self.assertAlmostEqual(
        loss_fn([labels[1]], [scores[1]], features).numpy(),
        _batch_aggregation(
            [_pairwise_loss(labels[1], scores[1], default_weights, loss_key)]),
        places=5)

    # Itemwise weights.
    loss_fn = ranking_losses.make_loss_fn(
        loss_key, weights_feature_name=weights_feature_name)
    features[weights_feature_name] = [itemwise_weights[0]]
    self.assertAlmostEqual(
        loss_fn([labels[0]], [scores[0]], features).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], itemwise_weights[0], loss_key)
        ]),
        places=5)

    features[weights_feature_name] = [itemwise_weights[1]]
    self.assertAlmostEqual(
        loss_fn([labels[1]], [scores[1]], features).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], itemwise_weights[1], loss_key)
        ]),
        places=5)

    # Multiple lists.
    features[weights_feature_name] = listwise_weights
    self.assertAlmostEqual(
        loss_fn(labels, scores, features).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], listwise_weights_expanded[0],
                           loss_key),
            _pairwise_loss(labels[1], scores[1], listwise_weights_expanded[1],
                           loss_key)
        ]),
        places=5)

    # Test LambdaWeight.
    lambda_weight = losses_impl.DCGLambdaWeight(
        rank_discount_fn=lambda r: 1. / tf.math.log1p(r), smooth_fraction=1.)
    loss_fn = ranking_losses.make_loss_fn(
        loss_key,
        weights_feature_name=weights_feature_name,
        lambda_weight=lambda_weight)
    self.assertAlmostEqual(
        loss_fn(labels, scores, features).numpy(),
        _batch_aggregation([
            _pairwise_loss(
                labels[0],
                scores[0],
                listwise_weights_expanded[0],
                loss_key,
                rank_discount_form='LOG'),
            _pairwise_loss(
                labels[1],
                scores[1],
                listwise_weights_expanded[1],
                loss_key,
                rank_discount_form='LOG')
        ]) * list_size,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        loss_key, reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        loss_key, reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_pairwise_hinge_loss(self):
    self._check_make_pairwise_loss(
        ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS)

  def test_make_pairwise_logistic_loss(self):
    self._check_make_pairwise_loss(
        ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS)

  def test_make_pairwise_soft_zero_one_loss(self):
    self._check_make_pairwise_loss(
        ranking_losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS)

  def test_make_pairwise_mse_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1.], [2.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.PAIRWISE_MSE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    expected = (((2. - 3.) - (1. - 0.))**2 + ((2. - 1.) - (1. - 0.))**2 +
                ((3. - 1.) - (0. - 0.))**2 + ((3. - 2.) - (2. - 0.))**2 +
                ((3. - 1.) - (2. - 0.))**2 + ((2. - 1.) - (0. - 0.))**2) / 6.

    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(), expected, places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.PAIRWISE_MSE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN,
        weights_feature_name=weights_feature_name)
    expected = (((2. - 3.) - (1. - 0.))**2 + ((2. - 1.) - (1. - 0.))**2 +
                ((3. - 1.) - (0. - 0.))**2 + 2 * ((3. - 2.) -
                                                  (2. - 0.))**2 + 2 *
                ((3. - 1.) - (2. - 0.))**2 + 2 * ((2. - 1.) -
                                                  (0. - 0.))**2) / 9.
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(), expected, places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.PAIRWISE_MSE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.PAIRWISE_MSE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_circle_loss(self):
    scores = [[0.1, 0.3, 0.2], [0.1, 0.2, 0.3]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.CIRCLE_LOSS)
    loss_0, _, _ = _circle_loss(labels[0], scores[0])
    loss_1, _, _ = _circle_loss(labels[1], scores[1])
    expected = (math.log1p(loss_0) + math.log1p(loss_1)) / 2.
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(), expected, places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.CIRCLE_LOSS,
        weights_feature_name=weights_feature_name)
    loss_0, _, _ = _circle_loss(labels[0], scores[0])
    loss_1, _, _ = _circle_loss(labels[1], scores[1])
    expected = (math.log1p(loss_0) * 2. + math.log1p(loss_1) * 1.) / 2.
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(), expected, places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.CIRCLE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.CIRCLE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_softmax_loss_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SOFTMAX_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -(math.log(_softmax(scores[0])[2]) +
          math.log(_softmax(scores[1])[2]) * 2.) / 2.,
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SOFTMAX_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(math.log(_softmax(scores[0])[2]) * 2. +
          math.log(_softmax(scores[1])[2]) * 2. * 1.) / 2.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SOFTMAX_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SOFTMAX_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_poly_one_softmax_loss_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 0., 2.]]
      weights = [[2.], [1.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: weights}
      with self.cached_session():
        loss_fn_simple = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.POLY_ONE_SOFTMAX_LOSS)
        self.assertAlmostEqual(
            loss_fn_simple(labels, scores, features).eval(),
            -(math.log(_softmax(scores[0])[2]) - 1 + _softmax(scores[0])[2] +
              (math.log(_softmax(scores[1])[2]) - 1 + _softmax(scores[1])[2]) *
              2.) / 2.,
            places=5)

        loss_fn_weighted = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.POLY_ONE_SOFTMAX_LOSS,
            weights_feature_name=weights_feature_name)
        self.assertAlmostEqual(
            loss_fn_weighted(labels, scores, features).eval(),
            -((math.log(_softmax(scores[0])[2]) - 1 + _softmax(scores[0])[2]) *
              2. +
              (math.log(_softmax(scores[1])[2]) - 1 + _softmax(scores[1])[2]) *
              2. * 1.) / 2.,
            places=5)

        # Test loss reduction method.
        # Two reduction methods should return different loss values.
        loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.POLY_ONE_SOFTMAX_LOSS,
            reduction=tf.compat.v1.losses.Reduction.SUM)
        loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.POLY_ONE_SOFTMAX_LOSS,
            reduction=tf.compat.v1.losses.Reduction.MEAN)
        self.assertNotAlmostEqual(
            loss_fn_1(labels, scores, features).eval(),
            loss_fn_2(labels, scores, features).eval())

  def test_make_unique_softmax_loss_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.UNIQUE_SOFTMAX_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -(math.log(_softmax(scores[0])[2]) + math.log(
            _softmax(scores[1][:2])[1]) + math.log(_softmax(scores[1])[2]) * 3.)
        / 2.,
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.UNIQUE_SOFTMAX_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(math.log(_softmax(scores[0])[2]) * 2. +
          math.log(_softmax(scores[1][:2])[1]) * 1. +
          math.log(_softmax(scores[1])[2]) * 3. * 1.) / 2.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.UNIQUE_SOFTMAX_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.UNIQUE_SOFTMAX_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_sigmoid_cross_entropy_loss_fn(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 1.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) +
         _sigmoid_cross_entropy(labels[1], scores[1])) / 6.,
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) * 2.0 +
         _sigmoid_cross_entropy(labels[1], scores[1])) / 6.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_mean_squared_loss_fn(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 1.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        (_mean_squared_error(labels[0], scores[0]) +
         _mean_squared_error(labels[1], scores[1])) / 6.,
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        (_mean_squared_error(labels[0], scores[0]) * 2.0 +
         _mean_squared_error(labels[1], scores[1])) / 6.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_list_mle_loss_fn(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.LIST_MLE_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) +
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)
    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.LIST_MLE_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(2 * (ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) + 1 *
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.LIST_MLE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.LIST_MLE_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_approx_ndcg_fn(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
        weights_feature_name=weights_feature_name,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)

    # Test different temperature.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
        params={'temperature': 10})
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
        params={'temperature': 0.01})
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_gumbel_approx_ndcg_fn(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [1., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    # sampled_scores = [[-1.7508768e-1, -4.6947412, -1.887345],
    #                   [-3.6629683e-1, -3.4472363, -1.2914587],
    #                   [-7.654705, -8.3514204, -7.1014347e-4],
    #                   [-10.080214, -8.7212124, -2.0500139e-4],
    #                   [-2.0658800e-1, -1.678545, -46.035358],
    #                   [-2.3852456e-1, -1.550176, -46.028168]]
    # sampled_rank = [[1, 3, 2],
    #                 [1, 3, 2],
    #                 [2, 3, 1],
    #                 [3, 2, 1],
    #                 [1, 2, 3],
    #                 [1, 2, 3]]
    # expanded_labels = [[0., 2., 1.], [0., 2., 1.],
    #                    [1., 0., 3.], [1., 0., 3.],
    #                    [1., 0., 0.], [1., 0., 0.]]
    # expanded_weights = [[2.], [2.], [1.], [1.], [1.], [1.]]

    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}

    tf.random.set_seed(1)
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.GUMBEL_APPROX_NDCG_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM,
        params={'temperature': 0.001},
        gumbel_params={
            'sample_size': 2,
            'seed': 1
        })
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -((2 / (3 / ln(2) + 1 / ln(3))) * (1 / ln(3) + 3 / ln(4)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4)) +
          (2 / (1 / ln(2))) * (1 / ln(2))),
        places=5)

    tf.random.set_seed(1)
    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.GUMBEL_APPROX_NDCG_LOSS,
        weights_feature_name=weights_feature_name,
        reduction=tf.compat.v1.losses.Reduction.SUM,
        params={'temperature': 0.001},
        gumbel_params={
            'sample_size': 2,
            'seed': 1
        })
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(2 * (2 / (3 / ln(2) + 1 / ln(3))) * (1 / ln(3) + 3 / ln(4)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4)) +
          (2 / (1 / ln(2))) * (1 / ln(2))),
        places=5)

  def test_make_approx_mrr_fn(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 0., 1.], [1., 0., 1.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -((1 / 2.) + 1 / 2. * (1 / 3. + 1 / 1.)),
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
        weights_feature_name=weights_feature_name,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(2 * 1 / 2. + 1 * 1 / 2. * (1 / 3. + 1 / 1.)),
        places=5)

    # Test different temperatures.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
        params={'temperature': 10})
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
        params={'temperature': 0.01})
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_neural_sort_cross_entropy_loss_fn(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 1.]]
    weights = [[2.], [1.]]
    p_scores = _neural_sort(scores)
    p_labels = _neural_sort(labels)
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_CROSS_ENTROPY_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        (_softmax_cross_entropy(p_labels[0], p_scores[0]) +
         _softmax_cross_entropy(p_labels[1], p_scores[1])) / 6.,
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_CROSS_ENTROPY_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        (_softmax_cross_entropy(p_labels[0], p_scores[0]) * 2.0 +
         _softmax_cross_entropy(p_labels[1], p_scores[1])) / 6.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_CROSS_ENTROPY_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_CROSS_ENTROPY_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_neural_sort_ndcg_fn(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_NDCG_LOSS,
        params={'temperature': 0.1},
        reduction=tf.compat.v1.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)

    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_NDCG_LOSS,
        params={'temperature': 0.1},
        weights_feature_name=weights_feature_name,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss_fn_weighted(labels, scores, features).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)

    # Test different temperatures.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_NDCG_LOSS,
        params={'temperature': 0.1})
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_NDCG_LOSS,
        params={'temperature': 100.})
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_NDCG_LOSS,
        reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.NEURAL_SORT_NDCG_LOSS,
        reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

  def test_make_loss_fn(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 1.]]
    weights = [[2.], [1.]]
    weights_1d = [2., 1.]
    weights_3d = [[[2.], [1.], [0.]], [[0.], [1.], [2.]]]
    weights_feature_name = 'weights'
    weights_1d_feature_name = 'weights_1d'
    weights_3d_feature_name = 'weights_3d'
    features = {
        weights_feature_name: weights,
        weights_1d_feature_name: weights_1d,
        weights_3d_feature_name: weights_3d
    }
    pairwise_hinge_loss = ranking_losses._pairwise_hinge_loss(labels,
                                                              scores).numpy()
    pairwise_hinge_loss_weighted = ranking_losses._pairwise_hinge_loss(
        labels, scores, weights=weights).numpy()
    pairwise_hinge_loss_itemwise_weighted = (
        ranking_losses._pairwise_hinge_loss(
            labels, scores, weights=tf.squeeze(weights_3d)).numpy())
    mean_squared_loss = ranking_losses._mean_squared_loss(labels,
                                                          scores).numpy()
    mean_squared_loss_weighted = ranking_losses._mean_squared_loss(
        labels, scores, weights=weights).numpy()
    mean_squared_loss_itemwise_weighted = ranking_losses._mean_squared_loss(
        labels, scores, weights=tf.squeeze(weights_3d)).numpy()

    loss_keys = [
        ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
        ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS
    ]
    loss_fn_simple = ranking_losses.make_loss_fn(loss_keys)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        pairwise_hinge_loss + mean_squared_loss,
        places=5)

    # With 2-d list-wise weighted examples.
    loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_weighted + mean_squared_loss_weighted,
        places=5)

    # With 1-d list-wise weighted examples.
    loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, weights_feature_name=weights_1d_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_weighted + mean_squared_loss_weighted,
        places=5)

    # With 3-d item-wise weighted examples.
    loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, weights_feature_name=weights_3d_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_itemwise_weighted +
        mean_squared_loss_itemwise_weighted,
        places=5)

    # With both weighted loss and weighted examples.
    loss_weights = [3., 2.]
    weighted_loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, loss_weights, weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        weighted_loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_weighted * loss_weights[0] +
        mean_squared_loss_weighted * loss_weights[1],
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        loss_keys, reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        loss_keys, reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

    # Test invalid inputs.
    with self.assertRaisesRegex(ValueError,
                                r'loss_keys cannot be None or empty.'):
      ranking_losses.make_loss_fn([])

    with self.assertRaisesRegex(ValueError,
                                r'loss_keys cannot be None or empty.'):
      ranking_losses.make_loss_fn('')

    with self.assertRaisesRegex(
        ValueError, r'loss_keys and loss_weights must have the same size.'):
      ranking_losses.make_loss_fn(loss_keys, [2.0])

    invalid_loss_fn = ranking_losses.make_loss_fn(['invalid_key'])
    with self.assertRaisesRegex(ValueError, r'Invalid loss_key: invalid_key.'):
      invalid_loss_fn(labels, scores, features).numpy()

  def test_make_loss_fn_with_a_string_of_keys_and_weights(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 1.]]
    weights = [[2.], [1.]]
    weights_1d = [2., 1.]
    weights_3d = [[[2.], [1.], [0.]], [[0.], [1.], [2.]]]
    weights_feature_name = 'weights'
    weights_1d_feature_name = 'weights_1d'
    weights_3d_feature_name = 'weights_3d'
    features = {
        weights_feature_name: weights,
        weights_1d_feature_name: weights_1d,
        weights_3d_feature_name: weights_3d
    }
    pairwise_hinge_loss = ranking_losses._pairwise_hinge_loss(labels,
                                                              scores).numpy()
    pairwise_hinge_loss_weighted = ranking_losses._pairwise_hinge_loss(
        labels, scores, weights=weights).numpy()
    pairwise_hinge_loss_itemwise_weighted = (
        ranking_losses._pairwise_hinge_loss(
            labels, scores, weights=tf.squeeze(weights_3d)).numpy())
    mean_squared_loss = ranking_losses._mean_squared_loss(labels,
                                                          scores).numpy()
    mean_squared_loss_weighted = ranking_losses._mean_squared_loss(
        labels, scores, weights=weights).numpy()
    mean_squared_loss_itemwise_weighted = ranking_losses._mean_squared_loss(
        labels, scores, weights=tf.squeeze(weights_3d)).numpy()

    loss_keys = 'pairwise_hinge_loss:1.0,mean_squared_loss:1.0'
    loss_fn_simple = ranking_losses.make_loss_fn(loss_keys)
    self.assertAlmostEqual(
        loss_fn_simple(labels, scores, features).numpy(),
        pairwise_hinge_loss + mean_squared_loss,
        places=5)

    # With 2-d list-wise weighted examples.
    loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_weighted + mean_squared_loss_weighted,
        places=5)

    # With 1-d list-wise weighted examples.
    loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, weights_feature_name=weights_1d_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_weighted + mean_squared_loss_weighted,
        places=5)

    # With 3-d item-wise weighted examples.
    loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys, weights_feature_name=weights_3d_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_itemwise_weighted +
        mean_squared_loss_itemwise_weighted,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        loss_keys, reduction=tf.compat.v1.losses.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        loss_keys, reduction=tf.compat.v1.losses.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(labels, scores, features).numpy(),
        loss_fn_2(labels, scores, features).numpy())

    # With both weighted loss and weighted examples.
    loss_keys_with_weights = 'pairwise_hinge_loss:3.0,mean_squared_loss:2.0'
    weighted_loss_fn_weighted_example = ranking_losses.make_loss_fn(
        loss_keys_with_weights, weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        weighted_loss_fn_weighted_example(labels, scores, features).numpy(),
        pairwise_hinge_loss_weighted * 3.0 + mean_squared_loss_weighted * 2.0,
        places=5)

    with self.assertRaisesRegex(
        ValueError,
        r'`loss_weights` has to be None when weights are encoded in `loss_keys`'
    ):
      ranking_losses.make_loss_fn(loss_keys, [2.0])

  def test_create_ndcg_lambda_weight(self):
    labels = [[2.0, 1.0]]
    ranks = [[1, 2]]
    lambda_weight = ranking_losses.create_ndcg_lambda_weight()
    scale = 2.
    max_dcg = 3.0 / math.log(2.) + 1.0 / math.log(3.)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks).numpy() / scale,
        [[[0., 2. * (1. / math.log(2.) - 1. / math.log(3.)) / max_dcg],
          [2. * (1. / math.log(2.) - 1. / math.log(3.)) / max_dcg, 0.]]])

  def test_create_reciprocal_rank_lambda_weight(self):
    labels = [[1.0, 2.0]]
    ranks = [[1, 2]]
    lambda_weight = ranking_losses.create_reciprocal_rank_lambda_weight()
    scale = 2.
    max_dcg = 2.5
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks).numpy() / scale,
        [[[0., 1. / 2. / max_dcg], [1. / 2. / max_dcg, 0.]]])

  def test_create_p_list_mle_lambda_weight(self):
    labels = [[1.0, 2.0]]
    ranks = [[1, 2]]
    lambda_weight = ranking_losses.create_p_list_mle_lambda_weight(2)
    self.assertAllClose(
        lambda_weight.individual_weights(labels, ranks).numpy(), [[1.0, 0.0]])


class LossMetricTest(tf.test.TestCase):

  def setUp(self):
    super(LossMetricTest, self).setUp()
    tf.compat.v1.reset_default_graph()

  def _check_metrics(self, metrics_and_values):
    """Checks metrics against values."""
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.local_variables_initializer())
      for (metric_op, update_op), value in metrics_and_values:
        sess.run(update_op)
        self.assertAlmostEqual(sess.run(metric_op), value, places=5)

  def test_make_loss_metric_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 0., 2.]]
      weights = [[2.], [1.]]
      weights_feature_name = 'weights'
      features = {
          weights_feature_name: weights,
      }
      m = ranking_losses.make_loss_metric_fn(
          ranking_losses.RankingLossKey.SOFTMAX_LOSS)
      m_w = ranking_losses.make_loss_metric_fn(
          ranking_losses.RankingLossKey.SOFTMAX_LOSS,
          weights_feature_name=weights_feature_name)
      self._check_metrics([
          (m(labels, scores,
             features), -(math.log(_softmax(scores[0])[2]) +
                          math.log(_softmax(scores[1])[2]) * 2.) / 3.),
          (m_w(labels, scores,
               features), -(math.log(_softmax(scores[0])[2]) * 2. +
                            math.log(_softmax(scores[1])[2]) * 2. * 1.) / 4.),
      ])

      # Value of loss metric is the same as loss with MEAN reduction.
      with self.cached_session():
        loss_fn_mean = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.SOFTMAX_LOSS,
            reduction=tf.compat.v1.losses.Reduction.MEAN)
        loss_mean = loss_fn_mean(labels, scores, features).eval()
      self._check_metrics([
          (m(labels, scores, features), loss_mean),
      ])


if __name__ == '__main__':
  tf.test.main()
