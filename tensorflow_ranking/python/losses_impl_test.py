# Copyright 2023 The TensorFlow Ranking Authors.
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

"""Tests for ranking losses implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_ranking.python import losses_impl


def ln(x):
  return math.log(x)


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


def _logodds_prob(logodds, alpha=1.0):
  return [[[math.exp(-alpha * (l - min(logodd[0])))
            for l in logodd[0]]]
          for logodd in logodds]


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


class UtilsTest(tf.test.TestCase):

  def test_approx_ranks(self):
    logits = [[100., 300., 200., 0.], [400., 200., 150., 300.]]
    target_ranks = [[3., 1., 2., 4.], [1., 3., 4., 2.]]

    approx_ranks = losses_impl.approx_ranks(logits)
    self.assertAllClose(approx_ranks, target_ranks)

  def test_inverse_max_dcg(self):
    labels = [[1., 4., 1., 0.], [4., 2., 0., 3.], [0., 0., 0., 0.]]
    target = [[0.04297], [0.033139], [0.]]
    target_1 = [[0.04621], [0.04621], [0.]]

    inverse_max_dcg = losses_impl.inverse_max_dcg(labels)
    inverse_max_dcg_1 = losses_impl.inverse_max_dcg(labels, topn=1)
    self.assertAllClose(inverse_max_dcg, target)
    self.assertAllClose(inverse_max_dcg_1, target_1)

  def test_ndcg(self):
    labels = [[1., 4., 1., 0.], [4., 2., 0., 3.], [0., 0., 0., 0.]]
    ranks = [[1, 2, 3, 4], [1, 3, 4, 2], [1, 2, 3, 4]]
    perm_mat = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    ndcg_ = [[0.679685], [0.95176], [0.]]
    ndcg_rank = [[0.679685], [1.], [0.]]
    ndcg_perm = [[0.679685], [1.], [0.]]

    ndcg = losses_impl.ndcg(labels)
    self.assertAllClose(ndcg, ndcg_)
    ndcg = losses_impl.ndcg(labels, ranks)
    self.assertAllClose(ndcg, ndcg_rank)
    ndcg = losses_impl.ndcg(labels, perm_mat=perm_mat)
    self.assertAllClose(ndcg, ndcg_perm)

  def test_gumbel_softmax_sample(self):
    tf.random.set_seed(1)
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 0., 1.], [1., 0., 1.], [0., 0., -1.]]
    labels_3d = [[[0., 0.], [0., 1.], [1., 0.]], [[1., 1.], [0., 0.], [0., 1.]],
                 [[0., 0.], [0., 0.], [-1., -1.]]]
    weights = [[2.], [1.], [1.]]

    sampled_scores = [[-1.7508768e-1, -4.6947412, -1.887345],
                      [-3.6629683e-1, -3.4472363, -1.2914587],
                      [-7.654705, -8.3514204, -7.1014347e-4],
                      [-10.080214, -8.7212124, -2.0500139e-4],
                      [-2.0658800e-1, -1.678545, -46.035358],
                      [-2.3852456e-1, -1.550176, -46.028168]]

    expanded_labels = [[0., 0., 1.], [0., 0., 1.], [1., 0., 1.], [1., 0., 1.],
                       [0., 0., -1.], [0., 0., -1.]]

    expanded_labels_3d = [[[0., 0.], [0., 1.], [1., 0.]],
                          [[0., 0.], [0., 1.], [1., 0.]],
                          [[1., 1.], [0., 0.], [0., 1.]],
                          [[1., 1.], [0., 0.], [0., 1.]],
                          [[0., 0.], [0., 0.], [-1., -1.]],
                          [[0., 0.], [0., 0.], [-1., -1.]]]

    expanded_weights = [[2.], [2.], [1.], [1.], [1.], [1.]]

    gumbel_sampler = losses_impl.GumbelSampler(sample_size=2, seed=1)
    gbl_labels, gbl_scores, gbl_weights = gumbel_sampler.sample(
        labels, scores, weights)
    self.assertAllEqual(gbl_labels, expanded_labels)
    self.assertAllClose(gbl_scores, sampled_scores, rtol=1e-3)
    self.assertAllEqual(gbl_weights, expanded_weights)

    tf.random.set_seed(1)
    gbl_labels_3d, gbl_scores, _ = gumbel_sampler.sample(
        labels_3d, scores, weights)
    self.assertAllEqual(gbl_labels_3d, expanded_labels_3d)
    self.assertAllClose(gbl_scores, sampled_scores, rtol=1e-3)

  def test_gumbel_softmax_ragged_sample(self):
    tf.random.set_seed(1)
    scores = tf.ragged.constant([[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2]])
    labels = tf.ragged.constant([[0., 0., 1.], [1., 0., 1.], [0., 0.]])
    weights = [[2.], [1.], [1.]]
    listwise_weights = tf.ragged.constant([[3., 1., 2.], [1., 1., 1.], [1.,
                                                                        2.]])

    sampled_scores = tf.ragged.constant(
        [[-1.7508768e-1, -4.6947412, -1.887345],
         [-3.6629683e-1, -3.4472363, -1.2914587],
         [-7.654705, -8.3514204, -7.1014347e-4],
         [-10.080214, -8.7212124, -2.0500139e-4], [-2.0658800e-1, -1.678545],
         [-2.3852456e-1, -1.550176]])

    expanded_labels = tf.ragged.constant([[0., 0., 1.], [0., 0., 1.],
                                          [1., 0., 1.], [1., 0., 1.], [0., 0.],
                                          [0., 0.]])

    expanded_weights = [[2.], [2.], [1.], [1.], [1.], [1.]]

    expanded_listwise_weights = tf.ragged.constant([[3., 1., 2.], [3., 1., 2.],
                                                    [1., 1., 1.], [1., 1., 1.],
                                                    [1., 2.], [1., 2.]])

    gumbel_sampler = losses_impl.GumbelSampler(
        sample_size=2, ragged=True, seed=1)
    gbl_labels, gbl_scores, gbl_weights = gumbel_sampler.sample(
        labels, scores, weights)
    self.assertAllEqual(gbl_labels, expanded_labels)
    self.assertAllClose(gbl_scores, sampled_scores, rtol=1e-3)
    self.assertAllEqual(gbl_weights, expanded_weights)

    tf.random.set_seed(1)
    gbl_labels, gbl_scores, gbl_weights = gumbel_sampler.sample(
        labels, scores, listwise_weights)
    self.assertAllEqual(gbl_labels, expanded_labels)
    self.assertAllClose(gbl_scores, sampled_scores, rtol=1e-3)
    self.assertAllEqual(gbl_weights, expanded_listwise_weights)

  def test_neural_sort(self):
    scores = [[140., -280., -40.], [0., 180., 1020.], [100., 120., -320.]]

    permuation_mat = [[[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                      [[0, 1, 0], [1, 0, 0], [0, 0, 1]]]

    smooth_perm = losses_impl.neural_sort(scores)
    self.assertAllClose(smooth_perm, permuation_mat, rtol=1e-3)

  def test_neural_sort_should_handle_mask(self):
    scores = [[3.0, 1.0, -1.0, 1000.0, 5.0, 2.0]]
    mask = [[True, True, True, False, False, True]]
    # Permutation matrix with two masked items sharing the last position.
    permutation_mat = [[[0.72140, 0.01321, 0.00000, 0., 0., 0.26539],
                        [0.21183, 0.21183, 0.00053, 0., 0., 0.57581],
                        [0.01204, 0.65723, 0.08895, 0., 0., 0.24178],
                        [0.00004, 0.11849, 0.87557, 0., 0., 0.0059],
                        [0., 0., 0., 0.5, 0.5, 0.], [0., 0., 0., 0.5, 0.5, 0.]]]

    smooth_perm = losses_impl.neural_sort(scores, mask=mask)
    self.assertAllClose(smooth_perm, permutation_mat, atol=1e-4)

  def test_gumbel_neural_sort(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]

    # sampled_scores = [[[-.291, -1.643, -2.826],
    #                    [-.0866, -2.924, -3.530]],
    #                   [[-12.42, -9.492, -7.939e-5],
    #                    [-8.859, -6.830, -1.223e-3]],
    #                   [[-.8930, -.5266, -45.80183],
    #                    [-.6650, -.7220, -45.94149]]]

    permuation_mat = [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                      [[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                       [[0, 0, 1], [0, 1, 0], [1, 0, 0]]],
                      [[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                       [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]]

    smooth_perm = losses_impl.gumbel_neural_sort(
        scores, sample_size=2, temperature=0.001, seed=1)
    self.assertAllClose(smooth_perm, permuation_mat, rtol=1e-3)


class LabelDiffLambdaWeightTest(tf.test.TestCase):
  """Test cases for LabelDiffLambdaWeight."""

  def test_default(self):
    """For the weight using rank diff."""
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    lambda_weight = losses_impl.LabelDiffLambdaWeight()
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks), [[
            [0., 1., 2.],
            [1., 0., 1.],
            [2., 1, 0.],
        ]])


class DCGLambdaWeightTest(tf.test.TestCase):
  """Test cases for DCGLambdaWeight."""

  def test_default(self):
    """For the weight using rank diff."""
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    scale = 3.
    lambda_weight = losses_impl.DCGLambdaWeight()
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1. / 2., 2. * 1. / 6.],
            [1. / 2., 0., 1. / 2.],
            [2. * 1. / 6., 1. / 2., 0.],
        ]])

  def test_smooth_fraction(self):
    """For the weights using absolute rank."""
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    scale = 3.
    lambda_weight = losses_impl.DCGLambdaWeight(smooth_fraction=1.0)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1. / 2., 2. * 2. / 3.],
            [1. / 2., 0., 1. / 6.],
            [2. * 2. / 3., 1. / 6., 0.],
        ]])

    lambda_weight = losses_impl.DCGLambdaWeight(topn=1, smooth_fraction=1.0)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1., 2.],
            [1., 0., 0.],
            [2., 0., 0.],
        ]])

  def test_topn(self):
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    scale = 3.
    lambda_weight = losses_impl.DCGLambdaWeight(topn=1)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1. / 2., 1. / 3.],
            [1. / 2., 0., 0.],
            [1. / 3., 0., 0.],
        ]])

  def test_invalid_labels(self):
    labels = [[2.0, 1.0, -1.0]]
    ranks = [[1, 2, 3]]
    scale = 3.
    lambda_weight = losses_impl.DCGLambdaWeight()
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1. / 2., 0.],
            [1. / 2., 0., 0.],
            [0., 0., 0.],
        ]])

  def test_gain_and_discount(self):
    labels = [[2.0, 1.0]]
    ranks = [[1, 2]]
    scale = 2.
    lambda_weight = losses_impl.DCGLambdaWeight(
        gain_fn=lambda x: tf.pow(2., x) - 1.,
        rank_discount_fn=lambda r: 1. / tf.math.log1p(r))
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 2. * (1. / math.log(2.) - 1. / math.log(3.))],
            [2. * (1. / math.log(2.) - 1. / math.log(3.)), 0.],
        ]])

  def test_normalized(self):
    labels = [[1.0, 2.0]]
    ranks = [[1, 2]]
    scale = 2.
    max_dcg = 2.5
    lambda_weight = losses_impl.DCGLambdaWeight(normalized=True)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1. / 2. / max_dcg],
            [1. / 2. / max_dcg, 0.],
        ]])

  def test_individual_weights(self):
    labels = [[1.0, 2.0]]
    ranks = [[1, 2]]
    max_dcg = 2.5
    lambda_weight = losses_impl.DCGLambdaWeight(normalized=True)
    self.assertAllClose(
        lambda_weight.individual_weights(labels, ranks), [
            [1. / max_dcg / 1., 2. / max_dcg / 2.],
        ])


class DCGLambdaWeightV2Test(tf.test.TestCase):
  """Test cases for DCGLambdaWeightV2."""

  def test_default(self):
    """For the weight using rank diff."""
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    scale = 3.
    lambda_weight = losses_impl.DCGLambdaWeightV2()
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1. / 2., 2. * 1. / 6.],
            [1. / 2., 0., 1. / 2.],
            [2. * 1. / 6., 1. / 2., 0.],
        ]])

  def test_topn(self):
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    scale = 3.
    lambda_weight = losses_impl.DCGLambdaWeightV2(topn=1)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks) / scale, [[
            [0., 1., 1. / 2.],
            [1., 0., 3. / 4.],
            [1. / 2., 3. / 4., 0.],
        ]])


class PrecisionLambdaWeightTest(tf.test.TestCase):
  """Test cases for PrecisionLambdaWeight."""

  def test_default(self):
    labels = [[2.0, 1.0, 0.0]]
    ranks = [[1, 2, 3]]
    lambda_weight = losses_impl.PrecisionLambdaWeight(topn=5)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks),
        [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])

    lambda_weight = losses_impl.PrecisionLambdaWeight(topn=1)
    self.assertAllClose(
        lambda_weight.pair_weights(labels, ranks),
        [[[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]]])


class LossesImplTest(tf.test.TestCase, parameterized.TestCase):

  def test_pointwise_compute_per_list(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    per_item_weights = [[2., 3., 4.], [1., 1., 1.]]

    # SigmoidCrossEntropyLoss is chosen as an arbitrary pointwise loss to
    # test the `compute_per_list` behavior.
    loss_fn = losses_impl.SigmoidCrossEntropyLoss(name=None)
    losses, weights = loss_fn.compute_per_list(labels, scores, per_item_weights)

    self.assertAllClose(losses, [1.3644443, 0.16292572])
    self.assertAllClose(weights, [2. + 3. + 4., 1. + 1. + 1.])

  def test_pairwise_compute_per_list(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    per_item_weights = [[2., 3., 4.], [1., 1., 1.]]

    # PairwiseHingeLoss is chosen as an arbitrary pairwise loss to test the
    # `compute_per_list` behavior.
    loss_fn = losses_impl.PairwiseHingeLoss(name=None)
    losses, weights = loss_fn.compute_per_list(labels, scores, per_item_weights)

    self.assertAllClose(losses, [1., 0.])
    self.assertAllClose(weights, [4. + 4., 1. + 1.])

  def test_listwise_compute_per_list(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    per_item_weights = [[2., 3., 4.], [1., 1., 1.]]

    # ApproxNDCGLoss is chosen as an arbitrary listwise loss to test the
    # `compute_per_list` behavior.
    loss_fn = losses_impl.ApproxNDCGLoss(name=None)
    losses, weights = loss_fn.compute_per_list(labels, scores, per_item_weights)

    self.assertAllClose(losses, [-0.63093, -0.796248])
    self.assertAllClose(weights, [4., 1.])

  @parameterized.parameters(
      (losses_impl.SigmoidCrossEntropyLoss, [1.3644443, -0.8190755], [9., 2.]),
      (losses_impl.MeanSquaredLoss, [3.6666667, 1.], [9., 2.]),
      (losses_impl.PairwiseHingeLoss, [1., 0.], [8., 1.]),
      (losses_impl.PairwiseLogisticLoss, [0.813262, 0.126928], [8., 1.]),
      (losses_impl.PairwiseSoftZeroOneLoss, [0.5, 0.119203], [8., 1.]),
      (losses_impl.ListMLELoss, [3.534534, 0.126928], [4., 1.]),
      (losses_impl.SoftmaxLoss, [1.407606, 0.126928], [4., 2.]),
      (losses_impl.UniqueSoftmaxLoss, [1.407606, 0.380784], [4., 1.]),
      (losses_impl.NeuralSortCrossEntropyLoss, [1.816267, 0.365334], [4., 1.]),
      (losses_impl.NeuralSortNDCGLoss, [-0.761571, -0.956006], [4., 1.]),
      (losses_impl.ApproxNDCGLoss, [-0.63093, -0.922917], [4., 1.]),
      (losses_impl.ApproxMRRLoss, [-0.5, -0.893493], [4., 1.]))
  def test_compute_per_list_with_ragged_tensors(self, loss_constructor,
                                                expected_losses,
                                                expected_weights):
    scores = tf.ragged.constant([[1., 3., 2.], [1., 3.]])
    labels = tf.ragged.constant([[0., 0., 1.], [0., 2.]])
    per_item_weights = tf.ragged.constant([[2., 3., 4.], [1., 1.]])

    tf.random.set_seed(42)
    loss_fn = loss_constructor(name=None, ragged=True)
    losses, weights = loss_fn.compute_per_list(labels, scores, per_item_weights)

    self.assertAllClose(losses, expected_losses)
    self.assertAllClose(weights, expected_weights)

  @parameterized.parameters(
      (losses_impl.SigmoidCrossEntropyLoss, [[1.313262, 3.048587, 0.126928],
                                             [1.313262, -2.951413, 0.]]),
      (losses_impl.MeanSquaredLoss, [[1., 9., 1.], [1., 1., 0.]]),
      (losses_impl.PairwiseHingeLoss, [[[0., 0., 0.], [0., 0., 0.], [
          0., 2., 0.
      ]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]),
      (losses_impl.PairwiseLogisticLoss, [[
          [0., 0., 0.], [0., 0., 0.], [0.313262, 1.313262, 0.]
      ], [[0., 0., 0.], [0.126928, 0., 0.], [0., 0., 0.]]]),
      (losses_impl.PairwiseSoftZeroOneLoss, [[
          [0., 0., 0.], [0., 0., 0.], [0.268941, 0.731059, 0.]
      ], [[0., 0., 0.], [0.11920291, 0., 0.], [0., 0., 0.]]]),
      (losses_impl.ListMLELoss, [[1.534534], [0.126928]]),
      (losses_impl.UniqueSoftmaxLoss, [[1.407606], [0.380784]]),
      (losses_impl.NeuralSortCrossEntropyLoss, [[1.816267], [0.365334]]),
      (losses_impl.NeuralSortNDCGLoss, [[-0.761571], [-0.956006]]),
      (losses_impl.ApproxNDCGLoss, [[-0.63093], [-0.922917]]),
      (losses_impl.ApproxMRRLoss, [[-0.5], [-0.893493]]))
  def test_compute_unreduced_loss_with_ragged_tensors(self, loss_constructor,
                                                      expected_weighted_losses):
    scores = tf.ragged.constant([[1., 3., 2.], [1., 3.]])
    labels = tf.ragged.constant([[0., 0., 1.], [0., 2.]])

    loss_fn = loss_constructor(name=None, ragged=True)
    losses, weights = loss_fn.compute_unreduced_loss(labels, scores)
    weighted_losses = tf.multiply(losses, weights)

    self.assertAllClose(weighted_losses, expected_weighted_losses)

  @parameterized.parameters(
      (losses_impl.SigmoidCrossEntropyLoss, [[2., 3., 4.], [1., 1., 0.]]),
      (losses_impl.MeanSquaredLoss, [[2., 3., 4.], [1., 1., 0.]]),
      (losses_impl.PairwiseHingeLoss, [[[2.], [3.], [4.]], [[1.], [1.], [0.]]]),
      (losses_impl.PairwiseLogisticLoss, [[[2.], [3.], [4.]], [[1.], [1.], [0.]]
                                         ]),
      (losses_impl.PairwiseSoftZeroOneLoss, [[[2.], [3.], [4.]],
                                             [[1.], [1.], [0.]]]),
      (losses_impl.ListMLELoss, [[4.], [1.]]),
      (losses_impl.SoftmaxLoss, [[4.], [1.]]),
      (losses_impl.UniqueSoftmaxLoss, [[4.], [1.]]),
      (losses_impl.NeuralSortCrossEntropyLoss, [[4.], [1.]]),
      (losses_impl.NeuralSortNDCGLoss, [[4.], [1.]]),
      (losses_impl.ApproxNDCGLoss, [[4.], [1.]]),
      (losses_impl.ApproxMRRLoss, [[4.], [1.]]))
  def test_normalize_weights_with_ragged_tensors(self, loss_constructor,
                                                 expected_weights):
    labels = tf.ragged.constant([[0., 0., 1.], [0., 2.]])
    per_item_weights = tf.ragged.constant([[2., 3., 4.], [1., 1.]])

    loss_fn = loss_constructor(name=None, ragged=True)
    weights = loss_fn.normalize_weights(labels, per_item_weights)

    self.assertAllClose(weights, expected_weights)


class PairwiseLogisticLossTest(tf.test.TestCase):

  def test_pairwise_logistic_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseLogisticLoss(name=None)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    logloss = lambda x: math.log(1. + math.exp(-x))
    expected = (logloss(3. - 2.) + logloss(1. - 2.) + logloss(3. - 1.) +
                logloss(3. - 2.)) / 4.
    self.assertAllClose(result, expected)

  def test_pairwise_logistic_loss_should_handle_per_list_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1.], [2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseLogisticLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    logloss = lambda x: math.log(1. + math.exp(-x))
    expected = (1. * (logloss(3. - 2.) + logloss(1. - 2.)) + 2. *
                (logloss(3. - 2.) + logloss(3. - 1.))) / 6.
    self.assertAllClose(result, expected)

  def test_pairwise_logistic_loss_should_handle_per_example_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1., 1., 2.], [1., 1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseLogisticLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    logloss = lambda x: math.log(1. + math.exp(-x))
    expected = ((2. * logloss(3. - 2.) + 2. * logloss(1. - 2.)) +
                (logloss(3. - 1.) + logloss(3. - 2.))) / 6.
    self.assertAllClose(result, expected)

  def test_pairwise_logistic_loss_should_handle_lambda_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN
    lambda_weight = losses_impl.DCGLambdaWeight()

    loss_fn = losses_impl.PairwiseLogisticLoss(
        name=None, lambda_weight=lambda_weight)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    logloss = lambda x: math.log(1. + math.exp(-x))
    expected = (((3. / 2.) * logloss(3. - 2.) + (3. / 2.) * logloss(1. - 2.)) +
                ((1. / 1.) * logloss(3. - 1.) +
                 (3. / 1.) * logloss(3. - 2.))) / ((3. / 2.) + (3. / 2.) +
                                                   (1. / 1.) + (3. / 1.))
    self.assertAllClose(result, expected)

  def test_pairwise_logistic_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseLogisticLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    logloss = lambda x: math.log(1. + math.exp(-x))
    expected = logloss(2. - 1.)
    self.assertAlmostEqual(result, expected, places=5)

  def test_pairwise_logistic_loss_should_handle_mask(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[1., 0., 0.], [0., 0., 2.]]
    mask = [[True, False, True], [True, True, True]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseLogisticLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask)

    logloss = lambda x: math.log(1. + math.exp(-x))
    expected = (logloss(1. - 2.) + logloss(3. - 1.) + logloss(3. - 2.)) / 3.
    self.assertAllClose(result, expected)


class PairwiseHingeLossTest(tf.test.TestCase):

  def test_pairwise_hinge_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseHingeLoss(name=None)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    hingeloss = lambda x: max(0, 1. - x)
    expected = (hingeloss(3. - 2.) + hingeloss(1. - 2.) + hingeloss(3. - 1.) +
                hingeloss(3. - 2.)) / 4.
    self.assertAllClose(result, expected)

  def test_pairwise_hinge_loss_should_handle_per_list_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1.], [2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseHingeLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    hingeloss = lambda x: max(0, 1. - x)
    expected = (1. * (hingeloss(3. - 2.) + hingeloss(1. - 2.)) + 2. *
                (hingeloss(3. - 2.) + hingeloss(3. - 1.))) / 6.
    self.assertAllClose(result, expected)

  def test_pairwise_hinge_loss_should_handle_per_example_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1., 1., 2.], [1., 1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseHingeLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    hingeloss = lambda x: max(0, 1. - x)
    expected = ((2. * hingeloss(3. - 2.) + 2. * hingeloss(1. - 2.)) +
                (hingeloss(3. - 1.) + hingeloss(3. - 2.))) / 6.
    self.assertAllClose(result, expected)

  def test_pairwise_hinge_loss_should_handle_lambda_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN
    lambda_weight = losses_impl.DCGLambdaWeight()

    loss_fn = losses_impl.PairwiseHingeLoss(
        name=None, lambda_weight=lambda_weight)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    hingeloss = lambda x: max(0, 1. - x)
    expected = (((3. / 2.) * hingeloss(3. - 2.) +
                 (3. / 2.) * hingeloss(1. - 2.)) +
                ((1. / 1.) * hingeloss(3. - 1.) +
                 (3. / 1.) * hingeloss(3. - 2.))) / ((3. / 2.) + (3. / 2.) +
                                                     (1. / 1.) + (3. / 1.))
    self.assertAllClose(result, expected)

  def test_pairwise_hinge_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseHingeLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    hingeloss = lambda x: max(0, 1. - x)
    expected = hingeloss(2. - 1.)
    self.assertAlmostEqual(result, expected, places=5)

  def test_pairwise_hinge_loss_should_handle_mask(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[1., 0., 0.], [0., 0., 2.]]
    mask = [[True, False, True], [True, True, True]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseHingeLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask)

    hingeloss = lambda x: max(0, 1. - x)
    expected = (hingeloss(1. - 2.) + hingeloss(3. - 1.) +
                hingeloss(3. - 2.)) / 3.
    self.assertAllClose(result, expected)


class PairwiseSoftZeroOneLossTest(tf.test.TestCase):

  def test_pairwise_soft_zero_one_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseSoftZeroOneLoss(name=None)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    softloss = lambda x: 1 / (1 + math.exp(x))
    expected = (softloss(3. - 2.) + softloss(1. - 2.) + softloss(3. - 1.) +
                softloss(3. - 2.)) / 4.
    self.assertAllClose(result, expected)

  def test_pairwise_soft_zero_one_loss_should_handle_per_list_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1.], [2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseSoftZeroOneLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    softloss = lambda x: 1 / (1 + math.exp(x))
    expected = (1. * (softloss(3. - 2.) + softloss(1. - 2.)) + 2. *
                (softloss(3. - 2.) + softloss(3. - 1.))) / 6.
    self.assertAllClose(result, expected)

  def test_pairwise_soft_zero_one_loss_should_handle_per_example_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1., 1., 2.], [1., 1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseSoftZeroOneLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    softloss = lambda x: 1 / (1 + math.exp(x))
    expected = ((2. * softloss(3. - 2.) + 2. * softloss(1. - 2.)) +
                (softloss(3. - 1.) + softloss(3. - 2.))) / 6.
    self.assertAllClose(result, expected)

  def test_pairwise_soft_zero_one_loss_should_handle_lambda_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN
    lambda_weight = losses_impl.DCGLambdaWeight()

    loss_fn = losses_impl.PairwiseSoftZeroOneLoss(
        name=None, lambda_weight=lambda_weight)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    softloss = lambda x: 1 / (1 + math.exp(x))
    expected = (((3. / 2.) * softloss(3. - 2.) +
                 (3. / 2.) * softloss(1. - 2.)) +
                ((1. / 1.) * softloss(3. - 1.) +
                 (3. / 1.) * softloss(3. - 2.))) / ((3. / 2.) + (3. / 2.) +
                                                    (1. / 1.) + (3. / 1.))
    self.assertAllClose(result, expected)

  def test_pairwise_soft_zero_one_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseSoftZeroOneLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    softloss = lambda x: 1 / (1 + math.exp(x))
    expected = softloss(2. - 1.)
    self.assertAlmostEqual(result, expected, places=5)

  def test_pairwise_soft_zero_one_loss_should_handle_mask(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[1., 0., 0.], [0., 0., 2.]]
    mask = [[True, False, True], [True, True, True]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseSoftZeroOneLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask)

    softloss = lambda x: 1 / (1 + math.exp(x))
    expected = (softloss(1. - 2.) + softloss(3. - 1.) + softloss(3. - 2.)) / 3.
    self.assertAllClose(result, expected)


class PairwiseMSELossTest(tf.test.TestCase):

  def test_pairwise_mse_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseMSELoss(name=None)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    expected = 2 * (((2. - 3.) - (1. - 0.))**2 + ((2. - 1.) - (1. - 0.))**2 +
                    ((3. - 1.) - (0. - 0.))**2 + ((3. - 2.) - (2. - 0.))**2 +
                    ((3. - 1.) - (2. - 0.))**2 + ((2. - 1.) -
                                                  (0. - 0.))**2) / (6. + 6.)
    self.assertAllClose(result, expected)

  def test_pairwise_mse_loss_should_handle_per_list_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1.], [2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseMSELoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    expected = 2 * (((2. - 3.) - (1. - 0.))**2 + ((2. - 1.) - (1. - 0.))**2 +
                    ((3. - 1.) - (0. - 0.))**2 + 2 *
                    ((3. - 2.) - (2. - 0.))**2 + 2 * ((3. - 1.) -
                                                      (2. - 0.))**2 + 2 *
                    ((2. - 1.) - (0. - 0.))**2) / (6. + 2. * 6.)
    self.assertAllClose(result, expected)

  def test_pairwise_mse_loss_should_handle_per_example_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [[1., 1., 2.], [1., 1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseMSELoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)
    # The per-example weights are applied asymmetrically. For example, the
    # first multiplier 3. is the weights[0][1] + weights[0][2].
    expected = ((3. * ((2. - 3.) - (1. - 0.))**2 + 3. *
                 ((2. - 1.) - (1. - 0.))**2 + 2. * ((3. - 1.) -
                                                    (0. - 0.))**2) + 2. *
                (((3. - 2.) - (2. - 0.))**2 + ((3. - 1.) - (2. - 0.))**2 +
                 ((2. - 1.) - (0. - 0.))**2)) / (8. + 6.)

    self.assertAllClose(result, expected)

  def test_pairwise_mse_loss_should_handle_lambda_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN
    lambda_weight = losses_impl.DCGLambdaWeight()

    loss_fn = losses_impl.PairwiseMSELoss(
        name=None, lambda_weight=lambda_weight)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    expected = ((3. / 2.) * ((2. - 3.) - (1. - 0.))**2 + (3. / 2.) *
                ((2. - 1.) - (1. - 0.))**2 + (3. / 1.) *
                ((3. - 2.) - (2. - 0.))**2 + (1. / 1.) *
                ((3. - 1.) - (2. - 0.))**2) / ((3. / 2.) + (3. / 2.) +
                                               (3. / 1.) + (1. / 1.))

    self.assertAllClose(result, expected)

  def test_pairwise_mse_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseMSELoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    expected = ((2. - 1.) - (1. - 0.))**2
    self.assertAlmostEqual(result, expected, places=5)

  def test_pairwise_mse_loss_should_handle_mask(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[1., 0., 0.], [0., 0., 2.]]
    mask = [[True, False, True], [True, True, True]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.PairwiseMSELoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask)

    expected = 2. * (((2. - 1.) - (0. - 1.))**2 + ((3. - 2.) - (2. - 0.))**2 +
                     ((3. - 1.) - (2. - 0.))**2 + ((2. - 1.) -
                                                   (0. - 0.))**2) / (2. + 6.)
    self.assertAllClose(result, expected)


class CircleLossTest(tf.test.TestCase):

  def test_circle_loss(self):
    scores = [[0.1, 0.3, 0.2], [0.1, 0.2, 0.3]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.CircleLoss(name=None)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    loss_0, _, _ = _circle_loss(labels[0], scores[0])
    loss_1, _, _ = _circle_loss(labels[1], scores[1])
    expected = (math.log1p(loss_0) + math.log1p(loss_1)) / 2
    self.assertAllClose(result, expected)

  def test_circle_loss_should_handle_per_list_weights(self):
    scores = [[0.1, 0.3, 0.2], [0.1, 0.2, 0.3]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1.], [2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.CircleLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    loss_0, _, _ = _circle_loss(labels[0], scores[0])
    loss_1, _, _ = _circle_loss(labels[1], scores[1])
    expected = (math.log1p(loss_0) * 1. + math.log1p(loss_1) * 2.) / 3.
    self.assertAllClose(result, expected)

  def test_circle_loss_should_handle_per_example_weights(self):
    scores = [[0.1, 0.3, 0.2], [0.1, 0.2, 0.3]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 1., 2.], [1., 1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.CircleLoss(name=None)
    result = loss_fn.compute(
        labels, scores, weights=weights, reduction=reduction)

    loss_0, _, _ = _circle_loss(labels[0], scores[0])
    loss_1, _, _ = _circle_loss(labels[1], scores[1])
    expected = (math.log1p(loss_0) * 2. + math.log1p(loss_1) * 1.) / 3.
    self.assertAllClose(result, expected)

  def test_circle_loss_should_handle_parameters(self):
    scores = [[.1, .3, .2], [.1, .2, .3]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.CircleLoss(name=None, gamma=4., margin=0.1)
    result = loss_fn.compute(labels, scores, weights=None, reduction=reduction)

    loss_0, _, _ = _circle_loss(labels[0], scores[0], gamma=4., margin=0.1)
    loss_1, _, _ = _circle_loss(labels[1], scores[1], gamma=4., margin=0.1)
    expected = (math.log1p(loss_0) + math.log1p(loss_1)) / 2
    self.assertAllClose(result, expected)

  def test_circle_loss_with_invalid_labels(self):
    scores = [[.1, .3, .2]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.CircleLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction)

    loss_0, _, _ = _circle_loss([0., 1.], [.1, .2])
    expected = math.log1p(loss_0)
    self.assertAllClose(result, expected)

  def test_circle_loss_should_handle_mask(self):
    scores = [[.1, .3, .2], [.1, .2, .3]]
    labels = [[1., 0., 0.], [0., 0., 2.]]
    mask = [[True, False, True], [True, True, True]]
    reduction = tf.compat.v1.losses.Reduction.MEAN

    loss_fn = losses_impl.CircleLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask)

    loss_0, _, _ = _circle_loss([1., 0.], [.1, .2])
    loss_1, _, _ = _circle_loss(labels[1], scores[1])
    expected = (math.log1p(loss_0) + math.log1p(loss_1)) / 2
    self.assertAllClose(result, expected)


class SoftmaxLossTest(tf.test.TestCase):

  def test_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.SoftmaxLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    self.assertAlmostEqual(
        result,
        -(math.log(_softmax(scores[0])[2]) +
          math.log(_softmax(scores[1])[2]) * 2.) / 2.,
        places=5)

  def test_softmax_loss_should_handle_per_example_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [1., 1., 2.], [0., 0., 0.]]
    example_weights = [[1., 1., 1.], [1., 2., 3.], [1., 0., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    probs = [_softmax(s) for s in scores]

    loss_fn = losses_impl.SoftmaxLoss(name=None)
    result = loss_fn.compute(labels, scores, example_weights, reduction).numpy()

    self.assertAlmostEqual(
        result,
        -(math.log(probs[0][2]) * 1. + math.log(probs[1][0]) * 1. * 1. +
          math.log(probs[1][1]) * 1. * 2. + math.log(probs[1][2]) * 2. * 3.) /
        2.,
        places=5)

  def test_softmax_loss_should_handle_per_list_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[1., 2., 1.], [0., 0., 2.], [0., 0., 0.]]
    list_weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    probs = [_softmax(s) for s in scores]

    loss_fn = losses_impl.SoftmaxLoss(name=None)
    result = loss_fn.compute(labels, scores, list_weights, reduction).numpy()

    self.assertAlmostEqual(
        result,
        -(math.log(probs[0][0]) * 1. * 2. + math.log(probs[0][1]) * 2. * 2. +
          math.log(probs[0][2]) * 1. * 2. + math.log(probs[1][2]) * 2. * 1.) /
        2.,
        places=5)

  def test_softmax_loss_should_support_lambda_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    lambda_weight = losses_impl.DCGLambdaWeight(
        rank_discount_fn=lambda r: 1. / tf.math.log1p(r))

    loss_fn = losses_impl.SoftmaxLoss(name=None, lambda_weight=lambda_weight)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    self.assertAlmostEqual(
        result,
        -(math.log(_softmax(scores[0])[2]) / math.log(1. + 2.) +
          math.log(_softmax(scores[1])[2]) * 2. / math.log(1. + 1.)) / 2.,
        places=5)

  def test_softmax_compute_per_list(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    per_item_weights = [[2., 3., 4.], [1., 1., 1.]]

    loss_fn = losses_impl.SoftmaxLoss(name=None)
    losses, weights = loss_fn.compute_per_list(labels, scores, per_item_weights)

    self.assertAllClose(losses, [1.407606, 0.407606])
    self.assertAllClose(weights, [4., 2.])

  def test_softmax_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.SoftmaxLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -(math.log(_softmax([1, 2])[1])),
        places=5)

  def test_softmax_loss_should_handle_mask(self):
    scores = [[1., 2., 3.]]
    labels = [[0., 1., 1.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.SoftmaxLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    self.assertAlmostEqual(result, -(math.log(_softmax([1, 3])[1])), places=5)

  def test_softmax_loss_should_handle_padded_labels_when_labels_are_zero(self):
    loss_fn = losses_impl.SoftmaxLoss(name=None)
    scores_padded = [[0., 0.]]
    labels_padded = [[0., -1.]]
    result_padded = loss_fn.compute_unreduced_loss(labels_padded,
                                                   scores_padded)[0]

    scores = [[0.]]
    labels = [[0.]]
    result = loss_fn.compute_unreduced_loss(labels, scores)[0]

    self.assertAllClose(result_padded, result)

  def test_softmax_loss_should_handle_fully_padded_labels(self):
    loss_fn = losses_impl.SoftmaxLoss(name=None)
    scores = [[0., 0.]]
    labels = [[-1., -1.]]
    result = loss_fn.compute_unreduced_loss(labels, scores)[0]

    self.assertAlmostEqual(result[0], 0.0)


class PolyOneSoftmaxLossTest(tf.test.TestCase):

  def test_poly_one_softmax_loss(self):
    with tf.Graph().as_default():
      with self.cached_session():
        scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
        labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
        reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

        loss_fn = losses_impl.PolyOneSoftmaxLoss(name=None, epsilon=3)
        result = loss_fn.compute(labels, scores, None, reduction).eval()

        self.assertAlmostEqual(
            result,
            -((math.log(_softmax(scores[0])[2]) - 3 *
               (1 - _softmax(scores[0])[2])) +
              (math.log(_softmax(scores[1])[2]) - 3 *
               (1 - _softmax(scores[1])[2])) * 2.) / 2.,
            places=5)


class UniqueSoftmaxLossTest(tf.test.TestCase):

  def test_unique_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.UniqueSoftmaxLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -(math.log(_softmax(scores[0])[2]) + math.log(
            _softmax(scores[1][:2])[1]) + math.log(_softmax(scores[1])[2]) * 3.)
        / 3.,
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        -(math.log(_softmax(scores[0])[2]) * 2. +
          math.log(_softmax(scores[1][:2])[1]) * 1. +
          math.log(_softmax(scores[1])[2]) * 3. * 1.) / 2.,
        places=5)

  def test_unique_softmax_compute_per_list(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    per_item_weights = [[2., 3., 4.], [1., 1., 1.]]

    loss_fn = losses_impl.UniqueSoftmaxLoss(name=None)
    losses, weights = loss_fn.compute_per_list(labels, scores, per_item_weights)

    self.assertAllClose(losses, [1.407606, 1.222818])
    self.assertAllClose(weights, [4., 1.])

  def test_unique_softmax_loss_should_handle_mask(self):
    scores = [[1., 2., 3., 2.]]
    labels = [[0., 1., 1., 0.]]
    mask = [[True, False, True, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.UniqueSoftmaxLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    self.assertAlmostEqual(
        result, -(math.log(_softmax([1, 3, 2])[1])), places=5)


class ListMLELossTest(tf.test.TestCase):

  def test_list_mle_loss(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    weights = [[2.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.ListMLELoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) +
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        -(2 * (ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) + 1 *
          (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)

  def test_list_mle_loss_tie(self):
    tf.random.set_seed(1)
    scores = [[0., ln(2), ln(3)]]
    labels = [[0., 0., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.ListMLELoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1))),
        places=5)

  def test_list_mle_loss_lambda_weight(self):
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    lw = losses_impl.ListMLELambdaWeight(
        rank_discount_fn=lambda rank: tf.pow(2., 3 - rank) - 1.)
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.ListMLELoss(name=None, lambda_weight=lw)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((3 * ln(3. / (3 + 2 + 1)) + 1 * ln(2. / (2 + 1)) + 0 * ln(1. / 1)) +
          (3 * ln(3. / (3 + 2 + 1)) + 1 * ln(1. / (1 + 2)) + 0 * ln(2. / 2))) /
        2,
        places=5)

  def test_list_mle_loss_should_handle_mask(self):
    tf.random.set_seed(1)
    scores = [[0., ln(2), ln(3)]]
    labels = [[0., 0., 1.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.ListMLELoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    self.assertAlmostEqual(result, -(ln(3. / (3 + 1)) + ln(1. / 1)), places=5)


class MeanSquaredLossTest(tf.test.TestCase):

  def test_mean_squared_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.MeanSquaredLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (_mean_squared_error(labels[0], scores[0]) + _mean_squared_error(
            labels[1], scores[1]) + _mean_squared_error(labels[2], scores[2])) /
        9.,
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        (_mean_squared_error(labels[0], scores[0]) * 2.0 + _mean_squared_error(
            labels[1], scores[1]) + _mean_squared_error(labels[2], scores[2])) /
        9.,
        places=5)

  def test_mean_squared_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.MeanSquaredLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(), (1. + 1.) / 2,
        places=5)

  def test_mean_squared_loss_should_handle_mask(self):
    scores = [[1., 3., 2.]]
    labels = [[0., 1., 1.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.MeanSquaredLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    self.assertAlmostEqual(result, (1. + 1.) / 2, places=5)


class SigmoidCrossEntropyLossTest(tf.test.TestCase):

  def test_sigmoid_cross_entropy_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.SigmoidCrossEntropyLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) * 2.0 +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)

  def test_sigmoid_cross_entropy_loss_with_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.SigmoidCrossEntropyLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (math.log(1. + math.exp(-2.)) + math.log(1. + math.exp(1))) / 2,
        places=5)

  def test_sigmoid_cross_entropy_loss_should_handle_mask(self):
    scores = [[1., 3., 2.]]
    labels = [[0., 1., 1.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.SigmoidCrossEntropyLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    self.assertAlmostEqual(
        result,
        (math.log(1. + math.exp(-2.)) + math.log(1. + math.exp(1.))) / 2.,
        places=5)


class OrdinalLossTest(tf.test.TestCase):

  def test_ordinal_loss(self):
    scores = [[[1., 2.], [3., 2.], [2., 3.]], [[1., 3.], [2., 2.], [3., 2.]],
              [[1., 1.], [2., 1.], [3., 3.]]]
    labels = [[0., 0., 1.], [0., 1., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.OrdinalLoss(name=None, ordinal_size=2)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (_sigmoid_cross_entropy([0., 0., 1.], [1., 3., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [2., 2., 3.]) +
         _sigmoid_cross_entropy([0., 1., 1.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 1.], [3., 2., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 1., 3.])) / 9.,
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        (_sigmoid_cross_entropy([0., 0., 1.], [1., 3., 2.]) * 2. +
         _sigmoid_cross_entropy([0., 0., 0.], [2., 2., 3.]) * 2. +
         _sigmoid_cross_entropy([0., 1., 1.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 1.], [3., 2., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 1., 3.])) / 9.,
        places=5)

  def test_ordinal_loss_with_invalid_labels(self):
    scores = [[[1., 1.], [3., 3.], [2., 2.]]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.OrdinalLoss(name=None, ordinal_size=2)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (_sigmoid_cross_entropy([0., 1.], [1., 2.]) +
         _sigmoid_cross_entropy([0., 0.], [1., 2.])) / 2.,
        places=5)

  def test_ordinal_loss_should_handle_mask(self):
    scores = [[[1., 1.], [2., 2.], [3., 3.], [2., 2.]]]
    labels = [[0., 1., 1., 0.]]
    mask = [[True, False, True, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.OrdinalLoss(name=None, ordinal_size=2)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    self.assertAlmostEqual(
        result,
        (_sigmoid_cross_entropy([0., 1., 0.], [1., 3., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 3., 2.])) / 3.,
        places=5)

  def test_ordinal_loss_with_fraction_label(self):
    scores = [[[1., 2.], [3., 2.], [2., 3.]], [[1., 3.], [2., 2.], [3., 2.]],
              [[1., 1.], [2., 1.], [3., 3.]]]
    labels = [[0., 0.5, 1.2], [0., 0.8, 2.], [0., 0., 0.1]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.OrdinalLoss(name=None, ordinal_size=2)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (_sigmoid_cross_entropy([0., 0., 1.], [1., 3., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [2., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 1.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 1.], [3., 2., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 1., 3.])) / 9.,
        places=5)
    loss_fn_frac = losses_impl.OrdinalLoss(name=None, ordinal_size=2,
                                           use_fraction_label=True)
    self.assertAlmostEqual(
        loss_fn_frac.compute(labels, scores, None, reduction).numpy(),
        (_sigmoid_cross_entropy([0., 0.5, 1.], [1., 3., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.2], [2., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0.8, 1.], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 1.], [3., 2., 2.]) +
         _sigmoid_cross_entropy([0., 0., 0.1], [1., 2., 3.]) +
         _sigmoid_cross_entropy([0., 0., 0.], [1., 1., 3.])) / 9.,
        places=5)


class MultiClassLossTest(tf.test.TestCase):

  def test_simple_ordinal_with_multi_class_loss(self):
    labels = [[0., 1., 2.]]
    # scores = [[1., 3., 2.]]
    # ord_boundaries = [0., 1.]
    prob = lambda x: 1. / (1. + math.exp(-x))
    softplus = lambda x: math.log(1. + math.exp(x))
    probs = [[[
        prob(0. - 1.),
        prob(softplus(1.) - 1.) - prob(0. - 1.), 1. - prob(softplus(1.) - 1.)
    ],
              [
                  prob(0. - 3.),
                  prob(softplus(1.) - 3.) - prob(0. - 3.),
                  1. - prob(softplus(1.) - 3.)
              ],
              [
                  prob(0. - 2.),
                  prob(softplus(1.) - 2.) - prob(0. - 2.),
                  1. - prob(softplus(1.) - 2.)
              ]]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    expected = (-math.log(probs[0][0][0]) - math.log(probs[0][1][1]) -
                math.log(probs[0][2][2])) / 3.

    loss_fn = losses_impl.MultiClassLoss(name=None, num_classes=3)
    self.assertAlmostEqual(
        loss_fn.compute(labels, probs, None, reduction).numpy(),
        expected,
        places=5)


class ClickEMLossTest(tf.test.TestCase):

  def test_click_em_loss(self):
    loss_fn = losses_impl.ClickEMLoss(name=None)
    loss_fn_weighted = losses_impl.ClickEMLoss(
        name='weighted', exam_loss_weight=2.0, rel_loss_weight=5.0)
    clicks = [[1., 0, 0, 0]]
    exam_logits = [[3., 3, 4, 100]]
    rel_logits = [[3., 2, 1, 100]]
    logits = tf.stack([exam_logits, rel_logits], axis=2)
    reduction = tf.compat.v1.losses.Reduction.SUM
    exam_given_clicks, rel_given_clicks = loss_fn._compute_latent_prob(
        clicks, exam_logits, rel_logits)
    self.assertAllClose(exam_given_clicks, [[1., 0.705384, 0.93624, 0.5]])
    self.assertAllClose(rel_given_clicks, [[1., 0.259496, 0.046613, 0.5]])

    self.assertAlmostEqual(
        loss_fn.compute(clicks, logits, None, reduction).numpy(),
        _sigmoid_cross_entropy([1., 0.705384, 0.93624, 0.5], exam_logits[0]) +
        _sigmoid_cross_entropy([1., 0.259496, 0.046613, 0.5], rel_logits[0]),
        places=5)

    # Test the loss weights.
    self.assertAlmostEqual(
        loss_fn_weighted.compute(clicks, logits, None, reduction).numpy(),
        _sigmoid_cross_entropy([1., 0.705384, 0.93624, 0.5], exam_logits[0]) *
        2.0 +
        _sigmoid_cross_entropy([1., 0.259496, 0.046613, 0.5], rel_logits[0]) *
        5.0,
        places=4)

  def test_click_em_loss_should_ignore_invalid_labels(self):
    clicks = [[1., -1., 0., 0.]]
    exam_logits = [[3., 3, 4, 100]]
    rel_logits = [[3., 2, 1, 100]]
    logits = tf.stack([exam_logits, rel_logits], axis=2)
    reduction = tf.compat.v1.losses.Reduction.SUM

    loss_fn = losses_impl.ClickEMLoss(name=None)
    result = loss_fn.compute(clicks, logits, None, reduction).numpy()

    self.assertAlmostEqual(
        result,
        _sigmoid_cross_entropy([1., 0.93624, 0.5], [3., 4., 100.]) +
        _sigmoid_cross_entropy([1., 0.046613, 0.5], [3., 1., 100.]),
        places=5)

  def test_click_em_loss_should_handle_mask(self):
    clicks = [[1., 1., 0., 0.]]
    exam_logits = [[3., 3., 4., 100.]]
    rel_logits = [[3., 2., 1., 100.]]
    mask = [[True, False, True, True]]
    logits = tf.stack([exam_logits, rel_logits], axis=2)
    reduction = tf.compat.v1.losses.Reduction.SUM

    loss_fn = losses_impl.ClickEMLoss(name=None)
    result = loss_fn.compute(clicks, logits, None, reduction, mask).numpy()

    self.assertAlmostEqual(
        result,
        _sigmoid_cross_entropy([1., 0.93624, 0.5], [3., 4., 100.]) +
        _sigmoid_cross_entropy([1., 0.046613, 0.5], [3., 1., 100.]),
        places=5)


class MixtureEMLossTest(tf.test.TestCase):

  def test_mixture_em_loss(self):
    loss_fn = losses_impl.MixtureEMLoss(name=None)
    clicks = [[1., 0, 0, 0]]
    model_1 = [[3., 3, 4, 100]]
    model_2 = [[3., 2, 1, 100]]
    logits = tf.stack([model_1, model_2], axis=2)
    reduction = tf.compat.v1.losses.Reduction.SUM
    logloss_1 = _sigmoid_cross_entropy(clicks[0], model_1[0])
    logloss_2 = _sigmoid_cross_entropy(clicks[0], model_2[0])
    model_probs = _logodds_prob([[[logloss_1, logloss_2]]])
    self.assertAllClose(
        loss_fn._compute_model_prob([[[logloss_1, logloss_2]]]), model_probs)

    self.assertAlmostEqual(
        loss_fn.compute(clicks, logits, None, reduction).numpy(),
        (logloss_1 * model_probs[0][0][0] + logloss_2 * model_probs[0][0][1]) /
        (model_probs[0][0][0] + model_probs[0][0][1]),
        places=4)

  def test_mixture_em_loss_should_ignore_invalid_labels(self):
    clicks = [[1., -1., 0., 0.]]
    model_1 = [[3., 3, 4, 100]]
    model_2 = [[3., 2, 1, 100]]
    logits = tf.stack([model_1, model_2], axis=2)
    reduction = tf.compat.v1.losses.Reduction.SUM
    logloss_1 = _sigmoid_cross_entropy([1., 0., 0.], [3., 4., 100.])
    logloss_2 = _sigmoid_cross_entropy([1., 0., 0.], [3., 1., 100.])
    model_probs = _logodds_prob([[[logloss_1, logloss_2]]])

    loss_fn = losses_impl.MixtureEMLoss(name=None)
    result = loss_fn.compute(clicks, logits, None, reduction).numpy()

    self.assertAlmostEqual(
        result,
        (logloss_1 * model_probs[0][0][0] + logloss_2 * model_probs[0][0][1]) /
        (model_probs[0][0][0] + model_probs[0][0][1]),
        places=5)

  def test_mixture_em_loss_should_handle_mask(self):
    clicks = [[1., 1., 0., 0.]]
    model_1 = [[3., 3., 4., 100.]]
    model_2 = [[3., 2., 1., 100.]]
    mask = [[True, False, True, True]]
    logits = tf.stack([model_1, model_2], axis=2)
    reduction = tf.compat.v1.losses.Reduction.SUM
    logloss_1 = _sigmoid_cross_entropy([1., 0., 0.], [3., 4., 100.])
    logloss_2 = _sigmoid_cross_entropy([1., 0., 0.], [3., 1., 100.])
    model_probs = _logodds_prob([[[logloss_1, logloss_2]]])

    loss_fn = losses_impl.MixtureEMLoss(name=None)
    result = loss_fn.compute(clicks, logits, None, reduction, mask).numpy()

    self.assertAlmostEqual(
        result,
        (logloss_1 * model_probs[0][0][0] + logloss_2 * model_probs[0][0][1]) /
        (model_probs[0][0][0] + model_probs[0][0][1]),
        places=5)


class ApproxNDCGLossTest(tf.test.TestCase):

  def test_approx_ndcg_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    # ranks= [[1,    3,    2],   [2,  1,   3],    [2,  1,    3]]
    labels = [[0., 2., 1.], [1., 0., -1.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    norm_weights = []
    for weight, label in zip(example_weights, labels):
      sum_label = sum(max(0, l) for l in label)
      norm_weights.append(
          sum(w * max(0, l) for w, l in zip(weight, label)) /
          sum_label if sum_label else 0)
    reduction = tf.compat.v1.losses.Reduction.SUM

    loss_fn = losses_impl.ApproxNDCGLoss(name=None, temperature=0.1)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + ln(2) *
          (1 / ln(3))),
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + 1 * ln(2) * (1 / ln(3))),
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, example_weights, reduction).numpy(),
        -(norm_weights[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
          (3 / ln(4) + 1 / ln(3)) + norm_weights[1] * ln(2) * (1 / ln(3))),
        places=5)

  def test_approx_ndcg_loss_should_handle_mask(self):
    scores = [[1., 3., 2.]]
    labels = [[0., 0., 1.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.ApproxNDCGLoss(name=None, temperature=1.)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    approxrank = 1. + 1. / (1. + math.exp(-(1. - 2.)))
    dcg = 1. / math.log(1. + approxrank)
    inv_max_dcg = math.log(1. + 1.)
    ndcg = dcg * inv_max_dcg
    self.assertAlmostEqual(result, -ndcg, places=5)

  def test_approx_ndcg_loss_should_handle_extreme_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., 0., 1000.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.ApproxNDCGLoss(name=None, temperature=1.)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    approxrank = 1. + 1. / (1. + math.exp(-(1. - 2.)))
    dcg = 1. / math.log(1. + approxrank)
    inv_max_dcg = math.log(1. + 1.)
    ndcg = dcg * inv_max_dcg
    self.assertAlmostEqual(result, -ndcg, places=5)


class ApproxMRRLossTest(tf.test.TestCase):

  def test_approx_mrr_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 0., 1.], [1., 0., 1.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM

    loss_fn = losses_impl.ApproxMRRLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((1 / 2.) + 1 / 2. * (1 / 3. + 1 / 1.)),
        places=5)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        -(2 * 1 / 2. + 1 * 1 / 2. * (1 / 3. + 1 / 1.)),
        places=5)

  def test_approx_mrr_loss_should_handle_mask(self):
    scores = [[1., 3., 2.]]
    labels = [[0., 0., 1.]]
    mask = [[True, False, True]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.ApproxMRRLoss(name=None, temperature=1.)
    result = loss_fn.compute(labels, scores, None, reduction, mask)

    approxrank = 1. + 1. / (1. + math.exp(-(1. - 2.)))
    self.assertAlmostEqual(result, -1. / approxrank, places=5)


class NeuralSortCrossEntropyLossTest(tf.test.TestCase):

  def test_neural_sort_cross_entropy_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., -3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    scores_valid = [[1.4, -2.8, -0.4], [0., 1.8, -1000.], [1., 1.2, -3.2]]
    labels_valid = [[0., 2., 1.], [1., 0., -1000.], [0., 0., 0.]]
    p_scores = _neural_sort(scores_valid)
    p_labels = _neural_sort(labels_valid)
    reduction = tf.compat.v1.losses.Reduction.SUM

    loss_fn = losses_impl.NeuralSortCrossEntropyLoss(name=None)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        (_softmax_cross_entropy(p_labels[0], p_scores[0]) / 3. +
         _softmax_cross_entropy(p_labels[1][0:2], p_scores[1][0:2]) / 2.),
        places=4)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        (_softmax_cross_entropy(p_labels[0], p_scores[0]) * 2.0 / 3. +
         _softmax_cross_entropy(p_labels[1][0:2], p_scores[1][0:2]) / 2.),
        places=4)

  def test_neural_sort_cross_entropy_loss_should_ignore_invalid_items(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.NeuralSortCrossEntropyLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()

    p_scores = _neural_sort([[1., 2.]])
    p_labels = _neural_sort([[0., 1.]])
    expected = _softmax_cross_entropy(p_labels[0], p_scores[0]) / 2.
    self.assertAlmostEqual(result, expected, places=5)

  def test_neural_sort_cross_entropy_loss_should_handle_mask(self):
    scores = [[2., 4., 3., 3., -1e10]]
    labels = [[0., 0., 1., 0., 1.]]
    mask = [[True, False, True, False, False]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.NeuralSortCrossEntropyLoss(name=None)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()

    p_scores = _neural_sort([[2., 3.]])
    p_labels = _neural_sort([[0., 1.]])
    expected = _softmax_cross_entropy(p_labels[0], p_scores[0]) / 2.
    self.assertAlmostEqual(result, expected, places=5)


class NeuralSortNDCGLossTest(tf.test.TestCase):

  def test_neural_sort_ndcg_loss(self):
    scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
    labels = [[0., 2., 1.], [1., 0., -3.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM

    loss_fn = losses_impl.NeuralSortNDCGLoss(name=None, temperature=0.1)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
          (1 / (1 / ln(2))) * (1 / ln(3))),
        places=4)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
          (1 / (1 / ln(2))) * (1 / ln(3))),
        places=4)

  def test_neural_sort_ndcg_loss_should_ignore_invalid_items(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.NeuralSortNDCGLoss(name=None, temperature=0.1)
    result = loss_fn.compute(labels, scores, None, reduction).numpy()
    self.assertAlmostEqual(result, -1., places=4)

  def test_neural_sort_ndcg_loss_should_handle_mask(self):
    scores = [[2., 4., 3., -5., 1000.0]]
    labels = [[0., 0., 1., 0., 1.]]
    mask = [[True, False, True, False, False]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    loss_fn = losses_impl.NeuralSortNDCGLoss(name=None, temperature=0.1)
    result = loss_fn.compute(labels, scores, None, reduction, mask).numpy()
    self.assertAlmostEqual(result, -1., places=4)


class CoupledRankDistilLossTest(tf.test.TestCase):

  def test_coupled_rank_distil_loss_basic(self):
    tf.random.set_seed(1)
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    # sampled_teacher_scores = [[-5.128768   -5.8270807  -0.00891006]]
    # [[-4.3828382  -4.4031367  -0.02503967]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.CoupledRankDistilLoss(name=None, sample_size=1)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((ln(2. / (2 + 1 + 3)) + ln(1. / (1 + 3)) + ln(3. / 3)) +
          (ln(3. / (3 + 1 + 2)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)

  def test_coupled_rank_distil_loss_handle_large_logits(self):
    tf.random.set_seed(1)
    scores = [[0., ln(3e30), ln(2e30)], [0., ln(2e30), ln(3e30)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.CoupledRankDistilLoss(name=None, sample_size=1)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((ln(2. / (2 + 1e-30 + 3)) + ln(1e-30 / (1e-30 + 3)) + ln(3. / 3)) +
          (ln(3. / (3 + 1e-30 + 2)) + ln(1e-30 / (1e-30 + 2)) + ln(2. / 2))) /
        2,
        places=5)

  def test_coupled_rank_distil_loss_with_weights(self):
    tf.random.set_seed(1)
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 2., 1.], [1., 0., 2.]]
    # sampled_teacher_scores = [[-5.128768   -5.8270807  -0.00891006]]
    # [[-4.3828382  -4.4031367  -0.02503967]]
    weights = [[2.], [1.]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.CoupledRankDistilLoss(name=None, sample_size=1)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, weights, reduction).numpy(),
        -(2 * (ln(2. / (2 + 1 + 3)) + ln(1. / (1 + 3)) + ln(3. / 3)) + 1 *
          (ln(3. / (3 + 1 + 2)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
        places=5)

  def test_coupled_rank_distil_loss_tie(self):
    tf.random.set_seed(1)
    scores = [[0., ln(2), ln(3)]]
    labels = [[0., 0., 1.]]
    # sampled_teacher_scores = [[-5.1262169e+00 -7.8245292e+00 -6.3590105e-03]].
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.CoupledRankDistilLoss(name=None, sample_size=1)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((ln(3. / (3 + 1 + 2)) + ln(1. / (1 + 2)) + ln(2. / 2))),
        places=5)

  def test_coupled_rank_distil_loss_should_ignore_invalid_items(self):
    tf.random.set_seed(1)
    scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
    labels = [[0., 1., -1.], [1., 0., 2.]]
    # sampled_teacher_scores = [[-5.128768   -5.8270807  -0.00891006]]
    # [[-4.3828382  -4.4031367  -0.02503967]]
    reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    loss_fn = losses_impl.CoupledRankDistilLoss(
        name=None, sample_size=1, topk=2)
    self.assertAlmostEqual(
        loss_fn.compute(labels, scores, None, reduction).numpy(),
        -((ln(1. / (1 + 3)) + ln(3. / 3)) +
          (ln(3. / (3 + 1 + 2)) + ln(1. / (1 + 2)))) / 2,
        places=5)

if __name__ == '__main__':
  tf.test.main()
