# Copyright 2021 The TensorFlow Ranking Authors.
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

# Lint as: python3
"""Tests for Keras losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import utils


def ln(x):
  return math.log(x)


def normalize_weights(weights, labels):
  sum_label = sum(max(0, l) for l in labels)
  return sum(w * max(0, l)
             for w, l in zip(weights, labels)) / sum_label if sum_label else 0


def _pairwise_loss(labels, scores, weights, loss_form, rank_discount_form=None):
  """Returns the pairwise loss given the loss form.

  Args:
    labels: A list of graded relevance.
    scores: A list of item ranking scores.
    weights: A list of item weights.
    loss_form: A string representing the form of loss.
    rank_discount_form: A string representing the form of rank discount.

  Returns:
    A tuple of (sum of loss, counts).
  """
  scores, labels, weights = zip(
      *sorted(zip(scores, labels, weights), reverse=True))

  def _rank_discount(rank_discount_form, rank):
    discount = {
        'LINEAR': 1. / rank,
        'LOG': 1. / ln(1. + rank),
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
        'hinge': max(0, 1 - score_diff) * delta,
        'logistic': ln(1. + math.exp(-score_diff)) * delta,
        'soft_zero_one': 1. / (1. + math.exp(score_diff)) * delta,
    }
    return loss_table[loss_form]

  loss = 0.
  count = 0.
  for i in range(len(labels)):
    count += 1.
    for j in range(len(labels)):
      if labels[i] > labels[j]:
        delta = _lambda_weight(labels[i] - labels[j], i, j)
        part_loss = _loss(scores[i] - scores[j], labels[i] - labels[j], delta)
        if weights[i] > 0:
          loss += part_loss * weights[i]
  return loss, count


def _batch_aggregation(batch_loss_list):
  """Returns the aggregated loss."""
  loss_sum = 0.
  weight_sum = 0.
  for loss, count in batch_loss_list:
    loss_sum += loss
    weight_sum += count
  return loss_sum / weight_sum


def _softmax(values):
  """Returns the softmax of `values`."""
  total = sum(math.exp(v) for v in values)
  return [math.exp(v) / total for v in values]


# Based on nn.sigmoid_cross_entropy_with_logits for x=logit and z=label the
# cross entropy is max(x, 0) - x * z + log(1 + exp(-abs(x))).
def _sigmoid_cross_entropy(labels, logits):

  def per_position_loss(logit, label):
    return max(logit, 0) - logit * label + ln(1 + math.exp(-abs(logit)))

  return sum(
      per_position_loss(logit, label) for label, logit in zip(labels, logits))


# Aggregates the per position squared error.
def _mean_squared_error(logits, labels):
  return sum((logit - label)**2 for label, logit in zip(labels, logits))


class LossesTest(parameterized.TestCase, tf.test.TestCase):

  def _check_pairwise_loss(self, loss_form):
    """Helper function to test `loss_fn`."""
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    listwise_weights = [[2.], [1.]]
    listwise_weights_expanded = [[2.] * 3, [1.] * 3]
    itemwise_weights = [[2., 3., 4.], [1., 1., 1.]]
    default_weights = [1.] * 3
    list_size = 3.
    loss_form_dict = {
        'hinge': losses.PairwiseHingeLoss(name='hinge'),
        'logistic': losses.PairwiseLogisticLoss(name='logistic'),
        'soft_zero_one': losses.PairwiseSoftZeroOneLoss(name='soft_zero_one'),
    }
    loss_fn = loss_form_dict[loss_form]

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
                sample_weight=[itemwise_weights[0]]).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], itemwise_weights[0], loss_form)
        ]),
        places=5)

    self.assertAlmostEqual(
        loss_fn([labels[1]], [scores[1]],
                sample_weight=[itemwise_weights[1]]).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], itemwise_weights[1], loss_form)
        ]),
        places=5)

    # Multiple lists.
    self.assertAlmostEqual(
        loss_fn(labels, scores, sample_weight=listwise_weights).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], listwise_weights_expanded[0],
                           loss_form),
            _pairwise_loss(labels[1], scores[1], listwise_weights_expanded[1],
                           loss_form)
        ]),
        places=5)

    # Test LambdaWeight.
    rank_discount_fn = lambda r: 1. / tf.math.log1p(r)
    lambda_weight = losses.DCGLambdaWeight(
        rank_discount_fn=rank_discount_fn, smooth_fraction=1.)
    loss_form_dict = {
        'hinge':
            losses.PairwiseHingeLoss(name='hinge', lambda_weight=lambda_weight),
        'logistic':
            losses.PairwiseLogisticLoss(
                name='logistic', lambda_weight=lambda_weight),
        'soft_zero_one':
            losses.PairwiseSoftZeroOneLoss(
                name='soft_zero_one', lambda_weight=lambda_weight),
    }
    loss_fn = loss_form_dict[loss_form]

    self.assertAlmostEqual(
        loss_fn(labels, scores, sample_weight=listwise_weights).numpy(),
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

  def test_pairwise_hinge_loss(self):
    self._check_pairwise_loss('hinge')

  def test_pairwise_logistic_loss(self):
    self._check_pairwise_loss('logistic')

  def test_pairwise_soft_zero_one_loss(self):
    self._check_pairwise_loss('soft_zero_one')

  def test_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.SoftmaxLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(ln(_softmax(scores[0])[2]) + ln(_softmax(scores[1])[2]) * 2.) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(ln(_softmax(scores[0])[2]) * 2. +
          ln(_softmax(scores[1])[2]) * 2. * 1.) / 3.,
        places=5)

    # Test LambdaWeight.
    rank_discount_fn = lambda r: 1. / tf.math.log1p(r)
    lambda_weight = losses.DCGLambdaWeight(rank_discount_fn=rank_discount_fn)
    loss = losses.SoftmaxLoss(lambda_weight=lambda_weight)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(ln(_softmax(scores[0])[2]) / ln(1. + 2.) +
          ln(_softmax(scores[1])[2]) * 2. / ln(1. + 1.)) / 3.,
        places=5)

  def test_unique_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.UniqueSoftmaxLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(ln(_softmax(scores[0])[2]) + ln(_softmax(scores[1][:2])[1]) +
          ln(_softmax(scores[1])[2]) * 3.) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(ln(_softmax(scores[0])[2]) * 2. + ln(_softmax(scores[1][:2])[1]) * 1.
          + ln(_softmax(scores[1])[2]) * 3. * 1.) / 3.,
        places=5)

  def test_click_em_loss(self):
    clicks = [[1., 0, 0, 0]]
    exam_logits = [[3., 3, 4, 100]]
    rel_logits = [[3., 2, 1, 100]]
    logits = tf.stack([exam_logits, rel_logits], axis=2)
    loss = losses.ClickEMLoss()
    self.assertAlmostEqual(
        loss(clicks, logits).numpy() * 4.,
        _sigmoid_cross_entropy([1., 0.705384, 0.93624, 0.5], exam_logits[0]) +
        _sigmoid_cross_entropy([1., 0.259496, 0.046613, 0.5], rel_logits[0]),
        places=5)

    loss = losses.ClickEMLoss(exam_loss_weight=2.0, rel_loss_weight=5.0)
    self.assertAlmostEqual(
        loss(clicks, logits).numpy() * 4.,
        _sigmoid_cross_entropy([1., 0.705384, 0.93624, 0.5], exam_logits[0]) *
        2.0 +
        _sigmoid_cross_entropy([1., 0.259496, 0.046613, 0.5], rel_logits[0]) *
        5.0,
        places=4)

  def test_sigmoid_cross_entropy_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.SigmoidCrossEntropyLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) * 2.0 +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)

  def test_mean_squared_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.MeanSquaredLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        (_mean_squared_error(labels[0], scores[0]) + _mean_squared_error(
            labels[1], scores[1]) + _mean_squared_error(labels[2], scores[2])) /
        9.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        (_mean_squared_error(labels[0], scores[0]) * 2.0 + _mean_squared_error(
            labels[1], scores[1]) + _mean_squared_error(labels[2], scores[2])) /
        9.,
        places=5)

  def test_list_mle_loss(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    weights = [[2.], [1.]]

    loss = losses.ListMLELoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) +
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * (ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) + 1 *
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)

  def test_list_mle_loss_tie(self):
    tf.random.set_seed(1)
    scores = [[0., ln(2), ln(3)]]
    labels = [[0., 0., 1.]]

    loss = losses.ListMLELoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1))),
        places=5)

  def test_list_mle_loss_lambda_weight(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    lw = losses.ListMLELambdaWeight(
        rank_discount_fn=lambda rank: tf.pow(2., 3 - rank) - 1.)

    loss = losses.ListMLELoss(lambda_weight=lw)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((3 * ln(3. / (3 + 2 + 1)) + 1 * ln(2. / (2 + 1)) + 0 * ln(1. / 1)) +
          (3 * ln(3. / (3 + 2 + 1)) + 1 * ln(1. / (1 + 2)) + 0 * ln(2. / 2))) /
        2,
        places=5)

  def test_approx_ndcg_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = [
        normalize_weights(w, l) for w, l in zip(example_weights, labels)
    ]

    loss = losses.ApproxNDCGLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, example_weights).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)

  def test_gumbel_approx_ndcg_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]

    # sampled_scores = [[-.291, -1.643, -2.826],
    #                   [-.0866, -2.924, -3.530],
    #                   [-12.42, -9.492, -7.939e-5],
    #                   [-8.859, -6.830, -1.223e-3],
    #                   [-.8930, -.5266, -45.80183],
    #                   [-.6650, -.7220, -45.94149]]
    # ranks    =     [[1,      2,      3],
    #                 [1,      2,      3],
    #                 [3,      2,      1],
    #                 [3,      2,      1],
    #                 [2,      1,      3],
    #                 [1,      2,      3]]
    # expanded_labels = [[0., 2., 1.],
    #                    [0., 2., 1.],
    #                    [1., 0., 3.],
    #                    [1., 0., 3.],
    #                    [0., 0., 0.],
    #                    [0., 0., 0.]]
    # expanded_weights = [[2.], [2.],
    #                     [1.], [1.],
    #                     [1.], [1.]]

    loss = losses.GumbelApproxNDCGLoss(sample_size=2, seed=1)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(3) + 1 / ln(4)) + 2 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 6,
        places=3)

  def test_gumbel_approx_ndcg_weighted_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.GumbelApproxNDCGLoss(sample_size=2, seed=1)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * 2 * (1 /
                   (3 / ln(2) + 1 / ln(3))) * (3 / ln(3) + 1 / ln(4)) + 1 * 2 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 6,
        places=3)

  def test_approx_ndcg_loss_sum(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = [
        normalize_weights(w, l) for w, l in zip(example_weights, labels)
    ]

    loss = losses.ApproxNDCGLoss(reduction=tf.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, example_weights).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)

  def test_approx_ndcg_loss_sum_batch(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = [
        normalize_weights(w, l) for w, l in zip(example_weights, labels)
    ]

    loss = losses.ApproxNDCGLoss(
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, example_weights).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)

  def test_approx_mrr_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 0., 1.], [1., 0., 1.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.ApproxMRRLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / 2.) + 1 / 2. * (1 / 3. + 1 / 1.)) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * 1 / 2. + 1 * 1 / 2. * (1 / 3. + 1 / 1.)) / 3.,
        places=5)

  def test_pairwise_logistic_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.PairwiseLogisticLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), ln(1 + math.exp(-1.)) / 3., places=5)

  def test_pairwise_logistic_loss_sum_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.PairwiseLogisticLoss(reduction=tf.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), ln(1 + math.exp(-1.)), places=5)

  def test_softmax_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.SoftmaxLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), -(ln(_softmax([1, 2])[1])), places=5)

  def test_unique_softmax_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.UniqueSoftmaxLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), -(ln(_softmax([1, 2])[1])), places=5)

  def test_sigmoid_cross_entropy_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.SigmoidCrossEntropyLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        (ln(1. + math.exp(-2.)) + ln(1. + math.exp(1))) / 3.,
        places=5)

  def test_mean_squared_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.MeanSquaredLoss()
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), (1. + 1.) / 3., places=5)

  @parameterized.parameters(
      (losses.PairwiseHingeLoss, 4.),
      (losses.PairwiseLogisticLoss, 2.9397852),
      (losses.PairwiseSoftZeroOneLoss, 1.7310586),
      (losses.SoftmaxLoss, 4.034129),
      (losses.UniqueSoftmaxLoss, 5.347391),
      (losses.SigmoidCrossEntropyLoss, 5.6642923),
      (losses.MeanSquaredLoss, 20.),
      (losses.ListMLELoss, 2.8477957),
      (losses.ApproxNDCGLoss, -1.2618682),
      (losses.ApproxMRRLoss, -1.0000114),
      (losses.GumbelApproxNDCGLoss, -12.72839))
  def test_loss_with_ragged_tensors(self, loss_constructor, expected):
    scores = tf.ragged.constant([[1., 3., 2.], [3., 2.]])
    labels = tf.ragged.constant([[0., 0., 1.], [0., 2.]])
    loss = loss_constructor(ragged=True,
                            reduction=tf.keras.losses.Reduction.SUM)

    result = loss(labels, scores)

    self.assertAlmostEqual(result.numpy(), expected, places=5)


class GetLossesTest(tf.test.TestCase):

  def _check_pairwise_loss(self, loss_form):
    """Helper function to test `loss_fn`."""
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    listwise_weights = [[2.], [1.]]
    listwise_weights_expanded = [[2.] * 3, [1.] * 3]
    itemwise_weights = [[2., 3., 4.], [1., 1., 1.]]
    default_weights = [1.] * 3
    list_size = 3.
    loss_form_dict = {
        'hinge':
            losses.get(
                loss=losses.RankingLossKey.PAIRWISE_HINGE_LOSS, name='hinge'),
        'logistic':
            losses.get(
                loss=losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
                name='logistic'),
        'soft_zero_one':
            losses.get(
                loss=losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS,
                name='soft_zero_one'),
    }
    loss_fn = loss_form_dict[loss_form]

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
                sample_weight=[itemwise_weights[0]]).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], itemwise_weights[0], loss_form)
        ]),
        places=5)

    self.assertAlmostEqual(
        loss_fn([labels[1]], [scores[1]],
                sample_weight=[itemwise_weights[1]]).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], itemwise_weights[1], loss_form)
        ]),
        places=5)

    # Multiple lists.
    self.assertAlmostEqual(
        loss_fn(labels, scores, sample_weight=listwise_weights).numpy(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], listwise_weights_expanded[0],
                           loss_form),
            _pairwise_loss(labels[1], scores[1], listwise_weights_expanded[1],
                           loss_form)
        ]),
        places=5)

    # Test LambdaWeight.
    rank_discount_fn = lambda r: 1. / tf.math.log1p(r)
    lambda_weight = losses.DCGLambdaWeight(
        rank_discount_fn=rank_discount_fn, smooth_fraction=1.)
    loss_form_dict = {
        'hinge':
            losses.get(
                loss=losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
                name='hinge',
                lambda_weight=lambda_weight),
        'logistic':
            losses.get(
                loss=losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
                name='logistic',
                lambda_weight=lambda_weight),
        'soft_zero_one':
            losses.get(
                loss=losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS,
                name='soft_zero_one',
                lambda_weight=lambda_weight),
    }
    loss_fn = loss_form_dict[loss_form]

    self.assertAlmostEqual(
        loss_fn(labels, scores, sample_weight=listwise_weights).numpy(),
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

  def test_pairwise_hinge_loss(self):
    self._check_pairwise_loss('hinge')

  def test_pairwise_logistic_loss(self):
    self._check_pairwise_loss('logistic')

  def test_pairwise_soft_zero_one_loss(self):
    self._check_pairwise_loss('soft_zero_one')

  def test_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.get(loss=losses.RankingLossKey.SOFTMAX_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(ln(_softmax(scores[0])[2]) + ln(_softmax(scores[1])[2]) * 2.) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(ln(_softmax(scores[0])[2]) * 2. +
          ln(_softmax(scores[1])[2]) * 2. * 1.) / 3.,
        places=5)

    # Test LambdaWeight.
    rank_discount_fn = lambda r: 1. / tf.math.log1p(r)
    lambda_weight = losses.DCGLambdaWeight(rank_discount_fn=rank_discount_fn)
    loss = losses.get(
        loss=losses.RankingLossKey.SOFTMAX_LOSS, lambda_weight=lambda_weight)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(ln(_softmax(scores[0])[2]) / ln(1. + 2.) +
          ln(_softmax(scores[1])[2]) * 2. / ln(1. + 1.)) / 3.,
        places=5)

  def test_unique_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.get(loss=losses.RankingLossKey.UNIQUE_SOFTMAX_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(ln(_softmax(scores[0])[2]) + ln(_softmax(scores[1][:2])[1]) +
          ln(_softmax(scores[1])[2]) * 3.) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(ln(_softmax(scores[0])[2]) * 2. + ln(_softmax(scores[1][:2])[1]) * 1.
          + ln(_softmax(scores[1])[2]) * 3. * 1.) / 3.,
        places=5)

  def test_sigmoid_cross_entropy_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.get(losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) * 2.0 +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)

  def test_mean_squared_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.get(loss=losses.RankingLossKey.MEAN_SQUARED_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        (_mean_squared_error(labels[0], scores[0]) + _mean_squared_error(
            labels[1], scores[1]) + _mean_squared_error(labels[2], scores[2])) /
        9.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        (_mean_squared_error(labels[0], scores[0]) * 2.0 + _mean_squared_error(
            labels[1], scores[1]) + _mean_squared_error(labels[2], scores[2])) /
        9.,
        places=5)

  def test_list_mle_loss(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    weights = [[2.], [1.]]

    loss = losses.get(loss=losses.RankingLossKey.LIST_MLE_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) +
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * (ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) + 1 *
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)

  def test_list_mle_loss_tie(self):
    tf.random.set_seed(1)
    scores = [[0., ln(2), ln(3)]]
    labels = [[0., 0., 1.]]

    loss = losses.get(loss=losses.RankingLossKey.LIST_MLE_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1))),
        places=5)

  def test_list_mle_loss_lambda_weight(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    lw = losses.ListMLELambdaWeight(
        rank_discount_fn=lambda rank: tf.pow(2., 3 - rank) - 1.)

    loss = losses.get(
        loss=losses.RankingLossKey.LIST_MLE_LOSS, lambda_weight=lw)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((3 * ln(3. / (3 + 2 + 1)) + 1 * ln(2. / (2 + 1)) + 0 * ln(1. / 1)) +
          (3 * ln(3. / (3 + 2 + 1)) + 1 * ln(1. / (1 + 2)) + 0 * ln(2. / 2))) /
        2,
        places=5)

  def test_approx_ndcg_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = [
        normalize_weights(w, l) for w, l in zip(example_weights, labels)
    ]

    loss = losses.get(loss=losses.RankingLossKey.APPROX_NDCG_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, example_weights).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)

  def test_gumbel_approx_ndcg_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]

    # sampled_scores = [[-.291, -1.643, -2.826],
    #                   [-.0866, -2.924, -3.530],
    #                   [-12.42, -9.492, -7.939e-5],
    #                   [-8.859, -6.830, -1.223e-3],
    #                   [-.8930, -.5266, -45.80183],
    #                   [-.6650, -.7220, -45.94149]]
    # ranks    =     [[1,      2,      3],
    #                 [1,      2,      3],
    #                 [3,      2,      1],
    #                 [3,      2,      1],
    #                 [2,      1,      3],
    #                 [1,      2,      3]]
    # expanded_labels = [[0., 2., 1.],
    #                    [0., 2., 1.],
    #                    [1., 0., 3.],
    #                    [1., 0., 3.],
    #                    [0., 0., 0.],
    #                    [0., 0., 0.]]
    # expanded_weights = [[2.], [2.],
    #                     [1.], [1.],
    #                     [1.], [1.]]

    loss = losses.get(
        loss=losses.RankingLossKey.GUMBEL_APPROX_NDCG_LOSS,
        sample_size=2,
        seed=1)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(3) + 1 / ln(4)) + 2 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 6,
        places=3)

  def test_gumbel_approx_ndcg_weighted_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.get(
        loss=losses.RankingLossKey.GUMBEL_APPROX_NDCG_LOSS,
        sample_size=2,
        seed=1)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * 2 * (1 /
                   (3 / ln(2) + 1 / ln(3))) * (3 / ln(3) + 1 / ln(4)) + 1 * 2 *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 6,
        places=3)

  def test_approx_ndcg_loss_sum(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = [
        normalize_weights(w, l) for w, l in zip(example_weights, labels)
    ]

    loss = losses.get(
        loss=losses.RankingLossKey.APPROX_NDCG_LOSS,
        reduction=tf.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, example_weights).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
        places=5)

  def test_approx_ndcg_loss_sum_batch(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = [
        normalize_weights(w, l) for w, l in zip(example_weights, labels)
    ]

    loss = losses.get(
        loss=losses.RankingLossKey.APPROX_NDCG_LOSS,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, example_weights).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] *
          (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))) / 3.,
        places=5)

  def test_approx_mrr_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 0., 1.], [1., 0., 1.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]

    loss = losses.get(loss=losses.RankingLossKey.APPROX_MRR_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        -((1 / 2.) + 1 / 2. * (1 / 3. + 1 / 1.)) / 3.,
        places=5)
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        -(2 * 1 / 2. + 1 * 1 / 2. * (1 / 3. + 1 / 1.)) / 3.,
        places=5)

    deserialized = tf.keras.utils.deserialize_keras_object(
        tf.keras.utils.serialize_keras_object(loss))
    self.assertAlmostEqual(
        loss(labels, scores, weights).numpy(),
        deserialized(labels, scores, weights).numpy())

  def test_pairwise_logistic_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.get(loss=losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), ln(1 + math.exp(-1.)) / 3., places=5)

  def test_pairwise_logistic_loss_sum_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.get(
        loss=losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
        reduction=tf.losses.Reduction.SUM)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), ln(1 + math.exp(-1.)), places=5)

  def test_softmax_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.get(loss=losses.RankingLossKey.SOFTMAX_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), -(ln(_softmax([1, 2])[1])), places=5)

  def test_sigmoid_cross_entropy_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.get(loss=losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(),
        (ln(1. + math.exp(-2.)) + ln(1. + math.exp(1))) / 3.,
        places=5)

  def test_mean_squared_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]

    loss = losses.get(loss=losses.RankingLossKey.MEAN_SQUARED_LOSS)
    self.assertAlmostEqual(
        loss(labels, scores).numpy(), (1. + 1.) / 3., places=5)


class SerializationTest(parameterized.TestCase, tf.test.TestCase):

  ndcg_lambda_weight = losses.NDCGLambdaWeight(
      gain_fn=utils.pow_minus_1, rank_discount_fn=utils.log2_inverse)

  def setUp(self):
    super().setUp()
    self._scores = [[1., 3., 2., 4.]]
    self._labels = [[0., -1., 2., 1.]]

  @parameterized.parameters(
      (losses.PairwiseHingeLoss(lambda_weight=ndcg_lambda_weight)),
      (losses.PairwiseLogisticLoss(lambda_weight=ndcg_lambda_weight)),
      (losses.PairwiseSoftZeroOneLoss(lambda_weight=ndcg_lambda_weight)),
      (losses.SoftmaxLoss(lambda_weight=ndcg_lambda_weight)),
      (losses.ListMLELoss(lambda_weight=ndcg_lambda_weight)),
      (losses.ApproxMRRLoss(lambda_weight=ndcg_lambda_weight)),
      (losses.ApproxNDCGLoss(lambda_weight=ndcg_lambda_weight)),
      (losses.ClickEMLoss(exam_loss_weight=2.0, rel_loss_weight=5.0)),
      (losses.SigmoidCrossEntropyLoss()),
      (losses.MeanSquaredLoss()),
      (losses.GumbelApproxNDCGLoss(seed=1)))
  def test_is_loss_serializable(self, loss):
    serialized = tf.keras.utils.serialize_keras_object(loss)
    deserialized = tf.keras.utils.deserialize_keras_object(serialized)
    self.assertDictEqual(loss.get_config(), deserialized.get_config())
    scores = self._scores
    if isinstance(loss, losses.ClickEMLoss):
      scores = tf.stack([scores, scores], axis=2)

    # Test whether the deserialized loss behaves the same as the original loss.
    # Note that we have to reset the random seed in between calls to make sure
    # results are reproducible for stochastic losses.
    tf.random.set_seed(0x4321)
    original_output = loss(self._labels, scores).numpy()
    tf.random.set_seed(0x4321)
    deserialized_output = deserialized(self._labels, scores).numpy()
    self.assertAllClose(original_output, deserialized_output)

  @parameterized.parameters(
      (losses.PairwiseHingeLoss(ragged=True)),
      (losses.PairwiseLogisticLoss(ragged=True)),
      (losses.PairwiseSoftZeroOneLoss(ragged=True)),
      (losses.SoftmaxLoss(ragged=True)),
      (losses.ListMLELoss(ragged=True)),
      (losses.ApproxMRRLoss(ragged=True)),
      (losses.ApproxNDCGLoss(ragged=True)),
      (losses.ClickEMLoss(ragged=True)),
      (losses.SigmoidCrossEntropyLoss(ragged=True)),
      (losses.MeanSquaredLoss(ragged=True)),
      (losses.GumbelApproxNDCGLoss(seed=1, ragged=True)))
  def test_is_ragged_loss_serializable(self, loss):
    scores = tf.ragged.constant([[1., 2., 4.], [0., 2.]])
    labels = tf.ragged.constant([[0., 2., 1.], [1., 0.]])

    if isinstance(loss, losses.ClickEMLoss):
      scores = tf.stack([scores, scores], axis=2)

    serialized = tf.keras.utils.serialize_keras_object(loss)
    deserialized = tf.keras.utils.deserialize_keras_object(serialized)

    # Test whether the deserialized loss behaves the same as the original loss.
    # Note that we have to reset the random seed in between calls to make sure
    # results are reproducible for stochastic losses.
    tf.random.set_seed(0x4321)
    original_output = loss(labels, scores).numpy()
    tf.random.set_seed(0x4321)
    deserialized_output = deserialized(labels, scores).numpy()
    self.assertAllClose(original_output, deserialized_output)

  @parameterized.parameters(
      (losses.DCGLambdaWeight()),
      (losses.DCGLambdaWeight(
          gain_fn=utils.identity, rank_discount_fn=utils.log2_inverse)),
      (losses.NDCGLambdaWeight()),
      (losses.NDCGLambdaWeight(
          gain_fn=utils.identity, rank_discount_fn=utils.log2_inverse)),
      (losses.PrecisionLambdaWeight()),
      (losses.PrecisionLambdaWeight(
          topn=5, positive_fn=utils.is_greater_equal_1)))
  def test_is_lambda_weight_serializable(self, lambda_weight):
    serialized = tf.keras.utils.serialize_keras_object(lambda_weight)
    deserialized = tf.keras.utils.deserialize_keras_object(serialized)
    self.assertDictEqual(lambda_weight.get_config(), deserialized.get_config())


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
