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

"""Tests for ranking metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import tensorflow as tf

from tensorflow_ranking.python import metrics as metrics_lib
from tensorflow_ranking.python import metrics_impl


def _dcg(label,
         rank,
         weight=1.0,
         gain_fn=lambda l: math.pow(2.0, l) - 1.0,
         rank_discount_fn=lambda r: 1. / math.log(r + 1.0, 2.0)):
  """Returns a single dcg addend.

  Args:
    label: The document label.
    rank: The document rank starting from 1.
    weight: The document weight.
    gain_fn: (function) Transforms labels.
    rank_discount_fn: (function) The rank discount function.

  Returns:
    A single dcg addend. e.g. weight*(2^relevance-1)/log2(rank+1).
  """
  return weight * gain_fn(label) * rank_discount_fn(rank)


def _alpha_dcg(label,
               cum_label,
               rank,
               weight=1.0,
               alpha=0.5,
               rank_discount_fn=lambda r: 1. / math.log(r + 1.0, 2.0)):
  """Returns a single alpha dcg addend.

  Args:
    label: The document label.
    cum_label: The cumulative document label.
    rank: The document rank starting from 1.
    weight: The document weight.
    alpha: The information gain about a subtopic from a doc.
    rank_discount_fn: (function) The rank discount function.

  Returns:
    A single alpha dcg addend. e.g.
    weight*(relevance*SUM((1-alpha)^SUM(relevance)))/log2(rank+1).
  """
  gain = sum(l * (1-alpha)**cl for l, cl in zip(label, cum_label))
  return weight * gain * rank_discount_fn(rank)


def _ap(relevances, scores, topn=None):
  """Returns the average precision (AP) of a single ranked list.

  The implementation here is copied from Equation (1.7) in
  Liu, T-Y "Learning to Rank for Information Retrieval" found at
  https://www.nowpublishers.com/article/DownloadSummary/INR-016

  Args:
    relevances: A `list` of document relevances, which are binary.
    scores: A `list` of document scores.
    topn: An `integer` specifying the number of items to be considered in the
      average precision computation.

  Returns:
    The MAP of the list as a float computed using the formula
    sum([P@k * rel for k, rel in enumerate(relevance)]) / sum(relevance)
    where P@k is the precision of the list at the cut off k.
  """

  def argsort(arr, reverse=True):
    arr_ind = sorted([(a, i) for i, a in enumerate(arr)], reverse=reverse)
    return list(zip(*arr_ind))[1]

  num_docs = len(relevances)
  if isinstance(topn, int) and topn > 0:
    num_docs = min(num_docs, topn)
  indices = argsort(scores)[:num_docs]
  ranked_relevances = [1. * relevances[i] for i in indices]
  precision = {}
  for k in range(1, num_docs + 1):
    precision[k] = sum(ranked_relevances[:k]) / k
  num_rel = sum(ranked_relevances[:num_docs])
  average_precision = sum(precision[k] * ranked_relevances[k - 1]
                          for k in precision) / num_rel if num_rel else 0
  return average_precision


def _label_boost(boost_form, label):
  """Returns the label boost.

  Args:
    boost_form: Either NDCG or PRECISION.
    label: The example label.

  Returns:
    A list of per list weight.
  """
  boost = {
      'NDCG': math.pow(2.0, label) - 1.0,
      'PRECISION': 1.0 if label >= 1.0 else 0.0,
      'MAP': 1.0 if label >= 1.0 else 0.0,
      'ALPHADCG': 1.0 if label >= 1.0 else 0.0,
  }
  return boost[boost_form]


def _example_weights_to_list_weights(weights, relevances, boost_form):
  """Returns list with per list weights derived from the per example weights.

  Args:
    weights: List of lists with per example weight.
    relevances:  List of lists with per example relevance score.
    boost_form: Either NDCG or PRECISION.

  Returns:
    A list of per list weight.
  """
  list_weights = []
  nonzero_relevance = 0.0
  for example_weights, labels in zip(weights, relevances):
    boosted_labels = [_label_boost(boost_form, label) for label in labels]
    numerator = sum((weight * boosted_labels[i])
                    for i, weight in enumerate(example_weights))
    denominator = sum(boosted_labels)
    if denominator == 0.0:
      list_weights.append(0.0)
    else:
      list_weights.append(numerator / denominator)
      nonzero_relevance += 1.0
  list_weights_sum = sum(list_weights)
  if list_weights_sum > 0.0:
    list_weights = [
        list_weights_sum / nonzero_relevance if w == 0.0 else w
        for w in list_weights
    ]

  return list_weights


class MetricsTest(tf.test.TestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()
    tf.compat.v1.reset_default_graph()

  def _check_metrics(self, metrics_and_values):
    """Checks metrics against values."""
    with self.test_session() as sess:
      sess.run(tf.compat.v1.local_variables_initializer())
      for (metric_op, update_op), value in metrics_and_values:
        sess.run(update_op)
        self.assertAlmostEqual(sess.run(metric_op), value, places=5)

  def test_reset_invalid_labels(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., -1., 1.]]
      labels, predictions, _, _ = metrics_impl._prepare_and_validate_params(
          labels, scores)
      self.assertAllClose(labels, [[0., 0., 1.]])
      self.assertAllClose(predictions, [[1., 1. - 1e-6, 2]])

  def test_mean_reciprocal_rank(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.], [3., 1., 2.]]
      # Note that scores are ranked in descending order.
      # ranks = [[3, 1, 2], [3, 2, 1], [1, 3, 2]]
      labels = [[0., 0., 1.], [0., 1., 2.], [0., 1., 0.]]
      # Note that the definition of MRR only uses the highest ranked
      # relevant item, where an item is relevant if its label is > 0.
      rel_rank = [2, 1, 3]
      weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
      mean_relevant_weights = [
          weights[0][2], sum(weights[1][1:]) / 2, weights[2][1]
      ]
      num_queries = len(scores)
      self.assertAlmostEqual(num_queries, 3)
      m = metrics_lib.mean_reciprocal_rank
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 1. / rel_rank[0]),
          (m([labels[0]], [scores[0]], topn=1), 0.),
          (m([labels[0]], [scores[0]], topn=2), 1. / rel_rank[0]),
          (m([labels[1]], [scores[1]]), 1. / rel_rank[1]),
          (m([labels[1]], [scores[1]], topn=1), 1. / rel_rank[1]),
          (m([labels[1]], [scores[1]], topn=6), 1. / rel_rank[1]),
          (m([labels[2]], [scores[2]]), 1. / rel_rank[2]),
          (m([labels[2]], [scores[2]], topn=1), 0.),
          (m([labels[2]], [scores[2]], topn=2), 0.),
          (m([labels[2]], [scores[2]], topn=3), 1. / rel_rank[2]),
          (m(labels[:2], scores[:2]), (0.5 + 1.0) / 2),
          (m(labels[:2], scores[:2], weights[:2]),
           (3. * 0.5 + (6. + 5.) / 2. * 1.) / (3. + (6. + 5) / 2.)),
          (m(labels,
             scores), sum([1. / rel_rank[ind] for ind in range(num_queries)]) /
           num_queries),
          (m(labels, scores,
             topn=1), sum([0., 1. / rel_rank[1], 0.]) / num_queries),
          (m(labels, scores, topn=2),
           sum([1. / rel_rank[0], 1. / rel_rank[1], 0.]) / num_queries),
          (m(labels, scores, weights),
           sum([
               mean_relevant_weights[ind] / rel_rank[ind]
               for ind in range(num_queries)
           ]) / sum(mean_relevant_weights)),
          (m(labels, scores, weights,
             topn=1), sum([0., mean_relevant_weights[1] / rel_rank[1], 0.]) /
           sum(mean_relevant_weights)),
      ])

  def test_make_mean_reciprocal_rank_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      # ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      # Note that the definition of MRR only uses the highest ranked
      # relevant item, where an item is relevant if its label is > 0.
      rel_rank = [2, 1]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      num_queries = len(scores)
      weights_feature_name = 'weights'
      features = {weights_feature_name: weights}
      m = metrics_lib.make_ranking_metric_fn(metrics_lib.RankingMetricKey.MRR)
      m_w = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.MRR,
          weights_feature_name=weights_feature_name)
      m_2 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.MRR, topn=1)
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), 0.5),
          (m(labels, scores, features), (0.5 + 1.0) / 2),
          (m_w(labels, scores, features),
           (3. * 0.5 + (6. + 5.) / 2. * 1.) / (3. + (6. + 5.) / 2.)),
          (m_2(labels, scores,
               features), (sum([0., 1. / rel_rank[1], 0.]) / num_queries)),
      ])

  def test_average_relevance_position(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      m = metrics_lib.average_relevance_position
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 2.),
          (m(labels, scores), (1. * 2. + 2. * 1. + 1. * 2.) / 4.),
          (m(labels, scores, weights),
           (3. * 1. * 2. + 6. * 2. * 1. + 5 * 1. * 2.) / (3. + 12. + 5.)),
      ])

  def test_make_average_relevance_position_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: weights}
      m = metrics_lib.make_ranking_metric_fn(metrics_lib.RankingMetricKey.ARP)
      m_w = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ARP,
          weights_feature_name=weights_feature_name)
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), 2.),
          (m(labels, scores, features), (1. * 2. + 2. * 1. + 1. * 2.) / 4.),
          (m_w(labels, scores, features),
           (3. * 1. * 2. + 6. * 2. * 1. + 5 * 1. * 2.) / (3. + 12. + 5.)),
      ])

  def test_precision(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      m = metrics_lib.precision
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 1. / 3.),
          (m([labels[0]], [scores[0]], topn=1), 0. / 1.),
          (m([labels[0]], [scores[0]], topn=2), 1. / 2.),
          (m(labels, scores), (1. / 3. + 2. / 3.) / 2.),
      ])

  def test_precision_with_zero_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 0.], [0., 1., 2.]]
      m = metrics_lib.precision
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 0.),
          (m(labels, scores), (0. + 2. / 3.) / 2.),
      ])

  def test_precision_with_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      list_weights = [[1.], [2.]]
      m = metrics_lib.precision
      as_list_weights = _example_weights_to_list_weights(
          weights, labels, 'PRECISION')
      self._check_metrics([
          (m(labels, scores,
             weights), ((1. / 3.) * as_list_weights[0] +
                        (2. / 3.) * as_list_weights[1]) / sum(as_list_weights)),
          (m(labels, scores, weights,
             topn=2), ((1. / 2.) * as_list_weights[0] +
                       (2. / 2.) * as_list_weights[1]) / sum(as_list_weights)),
          # Per list weight.
          (m(labels, scores,
             list_weights), ((1. / 3.) * list_weights[0][0] +
                             (2. / 3.) * list_weights[1][0]) / 3.0),
          # Zero precision case.
          (m(labels, scores, [[0., 0., 0.], [0., 0., 0.]], topn=2), 0.),
      ])

  def test_precision_weights_with_zero_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 0.], [0., 0., 1.], [0., 1., 2.]]
      weights = [[0., 0., 1.], [1., 2., 3.], [4., 5., 6.]]
      m = metrics_lib.precision
      as_list_weights = _example_weights_to_list_weights(
          weights, labels, 'PRECISION')
      self.assertAllClose(as_list_weights, [(3 + 5.5) / 2., 3, 5.5])
      self._check_metrics([
          (m(labels, scores, weights, topn=2), 0.0 * as_list_weights[0] +
           ((1. / 2.) * as_list_weights[1] +
            (2. / 2.) * as_list_weights[2]) / sum(as_list_weights)),
          (m(labels[0:2], scores[0:2], [[0., 0., 0.], [0., 0., 0.]]), 0.),
      ])

  def test_make_precision_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      features = {}
      m = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.PRECISION)
      m_top_1 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.PRECISION, topn=1)
      m_top_2 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.PRECISION, topn=2)
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), 1. / 3.),
          (m_top_1([labels[0]], [scores[0]], features), 0. / 1.),
          (m_top_2([labels[0]], [scores[0]], features), 1. / 2.),
          (m(labels, scores, features), (1. / 3. + 2. / 3.) / 2.),
      ])

  def test_mean_average_precision(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order, so the ranks are
      # [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      rels = [[0, 0, 1], [0, 1, 1]]
      m = metrics_lib.mean_average_precision
      self._check_metrics([
          (m([labels[0]], [scores[0]]), _ap(rels[0], scores[0])),
          (m([labels[0]], [scores[0]], topn=1), _ap(rels[0], scores[0],
                                                    topn=1)),
          (m([labels[0]], [scores[0]], topn=2), _ap(rels[0], scores[0],
                                                    topn=2)),
          (m(labels,
             scores), sum(_ap(rels[i], scores[i]) for i in range(2)) / 2.),
          (m(labels, scores, topn=1),
           sum(_ap(rels[i], scores[i], topn=1) for i in range(2)) / 2.),
      ])

  def test_mean_average_precision_with_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order, so the ranks are
      # [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      rels = [[0, 0, 1], [0, 1, 1]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      list_weights = [[1.], [2.]]
      m = metrics_lib.mean_average_precision
      as_list_weights = _example_weights_to_list_weights(weights, labels, 'MAP')
      # See Equation (1.7) in the following reference to make sense of
      # the formulas that appear in the following expression:
      # Liu, T-Y "Learning to Rank for Information Retrieval" found at
      # https://www.nowpublishers.com/article/DownloadSummary/INR-016
      self._check_metrics([
          (m([labels[0]], [scores[0]],
             [weights[0]]), ((1. / 2.) * 3.) / (0 * 1 + 0 * 2 + 1 * 3)),
          (m([labels[1]], [scores[1]], [weights[1]]),
           ((1. / 1.) * 6. + (2. / 2.) * 5.) / (0 * 4 + 1 * 5 + 1 * 6)),
          (m(labels, scores, weights),
           (((1. / 2.) * 3.) / (0 * 1 + 0 * 2 + 1 * 3) * as_list_weights[0] +
            ((1. / 1.) * 6. +
             (2. / 2.) * 5.) / (0 * 4 + 1 * 5 + 1 * 6) * as_list_weights[1]) /
           sum(as_list_weights)),
          (m(labels, scores, weights,
             topn=1), ((0 * as_list_weights[0] + ((1. / 1.) * 6.) /
                        (1 * 6) * as_list_weights[1]) / sum(as_list_weights))),
          (m(labels, scores, weights, topn=2),
           (((1. / 2.) * 3.) / (0 * 1 + 1 * 3) * as_list_weights[0] +
            ((1. / 1.) * 6. + (2. / 2.) * 5.) /
            (1 * 5 + 1 * 6) * as_list_weights[1]) / sum(as_list_weights)),
          # Per list weight.
          (m(labels, scores, list_weights),
           sum(_ap(rels[i], scores[i]) * list_weights[i][0] for i in range(2)) /
           sum(list_weights[i][0] for i in range(2))),
          # Zero precision case.
          (m(labels, scores, [0., 0., 0.], topn=2), 0.),
      ])

  def test_make_mean_average_precision_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order, so the ranks are
      # [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      rels = [[0, 0, 1], [0, 1, 1]]
      features = {}
      m = metrics_lib.make_ranking_metric_fn(metrics_lib.RankingMetricKey.MAP)
      m_top_1 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.MAP, topn=1)
      m_top_2 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.MAP, topn=2)
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), _ap(rels[0], scores[0])),
          (m_top_1([labels[0]], [scores[0]],
                   features), _ap(rels[0], scores[0], topn=1)),
          (m_top_2([labels[0]], [scores[0]],
                   features), _ap(rels[0], scores[0], topn=2)),
          (m(labels, scores,
             features), sum(_ap(rels[i], scores[i]) for i in range(2)) / 2.),
      ])

  def test_precision_ia(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [1., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      m = metrics_lib.precision_ia
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 1. / 3.),
          (m([labels[0]], [scores[0]], topn=1), 0. / 1.),
          (m([labels[0]], [scores[0]], topn=2), 1. / 2.),
          (m(labels, scores), (1. / 3. + 3. / 6.) / 2.),
      ])

  def test_precision_ia_with_zero_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      m = metrics_lib.precision_ia
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 0.),
          (m(labels, scores), (0. + 3. / 6.) / 2.),
      ])

  def test_precision_ia_with_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [1., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      sum_labels = [[0., 0., 1.], [0., 1., 1.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      list_weights = [[1.], [2.]]
      m = metrics_lib.precision_ia
      as_list_weights = _example_weights_to_list_weights(
          weights, sum_labels, 'PRECISION')
      self._check_metrics([
          (m(labels, scores,
             weights), ((1. / 3.) * as_list_weights[0] +
                        (3. / 6.) * as_list_weights[1]) / sum(as_list_weights)),
          (m(labels, scores, weights,
             topn=2), ((1. / 2.) * as_list_weights[0] +
                       (3. / 4.) * as_list_weights[1]) / sum(as_list_weights)),
          # Per list weight.
          (m(labels, scores,
             list_weights), ((1. / 3.) * list_weights[0][0] +
                             (3. / 6.) * list_weights[1][0]) / 3.0),
          # Zero precision case.
          (m(labels, scores, [[0., 0., 0.], [0., 0., 0.]], topn=2), 0.),
      ])

  def test_precision_ia_weights_with_zero_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [0., 0.], [1., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      sum_labels = [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.]]
      weights = [[0., 0., 1.], [1., 2., 3.], [4., 5., 6.]]
      m = metrics_lib.precision_ia
      as_list_weights = _example_weights_to_list_weights(
          weights, sum_labels, 'PRECISION')
      self.assertAllClose(as_list_weights, [(3 + 5.5) / 2., 3, 5.5])
      self._check_metrics([
          (m(labels, scores, weights, topn=2), 0.0 * as_list_weights[0] +
           ((1. / 2.) * as_list_weights[1] +
            (3. / 4.) * as_list_weights[2]) / sum(as_list_weights)),
          (m(labels[0:2], scores[0:2], [[0., 0., 0.], [0., 0., 0.]]), 0.),
      ])

  def test_make_precision_ia_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [1., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      features = {}
      m = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.PRECISION_IA)
      m_top_1 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.PRECISION_IA, topn=1)
      m_top_2 = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.PRECISION_IA, topn=2)
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), 1. / 3.),
          (m_top_1([labels[0]], [scores[0]], features), 0. / 1.),
          (m_top_2([labels[0]], [scores[0]], features), 1. / 2.),
          (m(labels, scores, features), (1. / 3. + 3. / 6.) / 2.),
      ])

  def test_normalized_discounted_cumulative_gain(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      m = metrics_lib.normalized_discounted_cumulative_gain
      expected_ndcg = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
          _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
      self._check_metrics([
          (m([labels[0]], [scores[0]]), expected_ndcg),
      ])
      expected_ndcg_1 = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
          _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
      expected_ndcg_2 = 1.0
      expected_ndcg = (expected_ndcg_1 + expected_ndcg_2) / 2.0
      self._check_metrics([
          (m(labels, scores), expected_ndcg),
      ])
      # Test different gain and discount functions.
      gain_fn = lambda rel: rel
      rank_discount_fn = lambda rank: 1. / rank

      def mod_dcg_fn(l, r):
        return _dcg(l, r, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)

      list_size = len(scores[0])
      ideal_labels = sorted(labels[0], reverse=True)
      list_dcgs = [
          mod_dcg_fn(labels[0][ind], ranks[0][ind]) for ind in range(list_size)
      ]
      ideal_dcgs = [
          mod_dcg_fn(ideal_labels[ind], ind + 1) for ind in range(list_size)
      ]
      expected_modified_ndcg_1 = sum(list_dcgs) / sum(ideal_dcgs)
      self._check_metrics([
          (m([labels[0]], [scores[0]],
             gain_fn=gain_fn,
             rank_discount_fn=rank_discount_fn), expected_modified_ndcg_1),
      ])

  def test_normalized_discounted_cumulative_gain_with_zero_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 0.], [0., 1., 2.]]

      m = metrics_lib.normalized_discounted_cumulative_gain
      self._check_metrics([(m(labels, scores), (0. + 1.) / 2.0)])

  def test_normalized_discounted_cumulative_gain_with_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      list_weights = [[1.], [2.]]
      m = metrics_lib.normalized_discounted_cumulative_gain
      self._check_metrics([
          (m([labels[0]], [scores[0]], weights[0],
             topn=1), _dcg(0., 1, 2.) / _dcg(1., 1, 3.)),
          (m([labels[0]], [scores[0]], weights[0]),
           (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
           (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
      ])
      expected_ndcg_1 = (
          _dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) / (
              _dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))
      expected_ndcg_2 = 1.0
      as_list_weights = _example_weights_to_list_weights(
          weights, labels, 'NDCG')
      expected_ndcg = (expected_ndcg_1 * as_list_weights[0] + expected_ndcg_2 *
                       as_list_weights[1]) / sum(as_list_weights)
      self._check_metrics([
          (m(labels, scores, weights), expected_ndcg),
      ])
      expected_ndcg_1 = _dcg(0., 1, 2.) / _dcg(1., 1, 3.)
      expected_ndcg_2 = 1.0
      expected_ndcg = (expected_ndcg_1 * as_list_weights[0] + expected_ndcg_2 *
                       as_list_weights[1]) / sum(as_list_weights)
      self._check_metrics([
          (m(labels, scores, weights, topn=1), expected_ndcg),
      ])
      expected_ndcg_1 = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
          _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
      expected_ndcg_2 = 1.0
      expected_ndcg = (expected_ndcg_1 + 2. * expected_ndcg_2) / 3.0
      self._check_metrics([(m(labels, scores, list_weights), expected_ndcg)])
      # Test zero NDCG cases.
      self._check_metrics([
          (m(labels, scores, [[0.], [0.]]), 0.),
          (m([[0., 0., 0.]], [scores[0]], weights[0], topn=1), 0.),
      ])

  def test_normalized_discounted_cumulative_gain_with_weights_zero_relevance(
      self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 0.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      m = metrics_lib.normalized_discounted_cumulative_gain
      expected_ndcg_1 = 0.0
      expected_ndcg_2 = 1.0
      as_list_weights = _example_weights_to_list_weights(
          weights, labels, 'NDCG')
      self.assertAllClose(as_list_weights, [5.75, 5.75])
      expected_ndcg = (expected_ndcg_1 * as_list_weights[0] + expected_ndcg_2 *
                       as_list_weights[1]) / sum(as_list_weights)
      self._check_metrics([
          (m(labels, scores, weights), expected_ndcg),
      ])
      # Test zero NDCG cases.
      self._check_metrics([(m(labels, scores, [[0.], [0.]]), 0.)])

  def test_normalized_discounted_cumulative_gain_with_zero_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      m = metrics_lib.normalized_discounted_cumulative_gain
      self._check_metrics([
          (m(labels, scores, [[0.], [0.]]), 0.),
          (m([[0., 0., 0.]], [scores[0]], weights[0], topn=1), 0.),
      ])

  def test_make_normalized_discounted_cumulative_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      weights_3d = [[[1.], [2.], [3.]], [[4.], [5.], [6.]]]
      list_weights = [1., 0.]
      list_weights_2d = [[1.], [0.]]
      weights_feature_name = 'weights'
      weights_invalid_feature_name = 'weights_invalid'
      weights_3d_feature_name = 'weights_3d'
      list_weights_name = 'list_weights'
      list_weights_2d_name = 'list_weights_2d'
      features = {
          weights_feature_name: [weights[0]],
          weights_invalid_feature_name: weights[0],
          weights_3d_feature_name: [weights_3d[0]],
          list_weights_name: list_weights,
          list_weights_2d_name: list_weights_2d
      }
      m = metrics_lib.make_ranking_metric_fn(metrics_lib.RankingMetricKey.NDCG)

      expected_ndcg = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
          _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), expected_ndcg),
      ])
      expected_ndcg_1 = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
          _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
      expected_ndcg_2 = 1.0
      expected_ndcg = (expected_ndcg_1 + expected_ndcg_2) / 2.0
      self._check_metrics([
          (m(labels, scores, features), expected_ndcg),
      ])

      # With item-wise weights.
      m_top = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.NDCG,
          weights_feature_name=weights_feature_name,
          topn=1)
      m_weight = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.NDCG,
          weights_feature_name=weights_feature_name)
      m_weights_3d = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.NDCG,
          weights_feature_name=weights_3d_feature_name)
      self._check_metrics([
          (m_top([labels[0]], [scores[0]],
                 features), _dcg(0., 1, 2.) / _dcg(1., 1, 3.)),
          (m_weight([labels[0]], [scores[0]], features),
           (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
           (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
          (m_weights_3d([labels[0]], [scores[0]], features),
           (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
           (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
      ])
      with self.assertRaises(ValueError):
        m_weight_invalid = metrics_lib.make_ranking_metric_fn(
            metrics_lib.RankingMetricKey.NDCG,
            weights_feature_name=weights_invalid_feature_name)
        m_weight_invalid([labels[0]], [scores[0]], features)

      # With list-wise weights.
      m_list_weight = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.NDCG,
          weights_feature_name=list_weights_name)
      m_list_weight_2d = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.NDCG,
          weights_feature_name=list_weights_2d_name)
      self._check_metrics([
          (m_list_weight(labels, scores, features),
           (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
           (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
          (m_list_weight_2d(labels, scores, features),
           (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
           (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
      ])

      # Test different gain and discount functions.
      gain_fn = lambda rel: rel
      rank_discount_fn = lambda rank: 1. / rank

      def mod_dcg_fn(l, r):
        return _dcg(l, r, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)

      m_mod = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.NDCG,
          gain_fn=gain_fn,
          rank_discount_fn=rank_discount_fn)
      list_size = len(scores[0])
      expected_modified_dcg_1 = sum([
          mod_dcg_fn(labels[0][ind], ranks[0][ind]) for ind in range(list_size)
      ])
      self._check_metrics([
          (m_mod([labels[0]], [scores[0]], features), expected_modified_dcg_1),
      ])

  def test_discounted_cumulative_gain(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 1., 1.], [2., 2., 1.]]
      m = metrics_lib.discounted_cumulative_gain
      expected_dcg_1 = _dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)
      self._check_metrics([
          (m([labels[0]], [scores[0]]), expected_dcg_1),
      ])
      expected_dcg_2 = _dcg(2., 1) + _dcg(1., 2)
      expected_dcg_2_weighted = _dcg(2., 1) + _dcg(1., 2) * 2.
      expected_weight_2 = ((4 - 1) * 1. + (2 - 1) * 2.) / (4 - 1 + 2 - 1)
      self._check_metrics([
          (m(labels, scores), (expected_dcg_1 + expected_dcg_2) / 2.0),
          (m(labels, scores,
             weights), (expected_dcg_1 + expected_dcg_2_weighted) /
           (1. + expected_weight_2)),
      ])
      # Test different gain and discount functions.
      gain_fn = lambda rel: rel
      rank_discount_fn = lambda rank: 1. / rank

      def mod_dcg_fn(l, r):
        return _dcg(l, r, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)

      list_size = len(scores[0])
      expected_modified_dcg_1 = sum([
          mod_dcg_fn(labels[0][ind], ranks[0][ind]) for ind in range(list_size)
      ])
      self._check_metrics([
          (m([labels[0]], [scores[0]],
             gain_fn=gain_fn,
             rank_discount_fn=rank_discount_fn), expected_modified_dcg_1),
      ])

  def test_make_discounted_cumulative_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 1., 1.], [2., 2., 1.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: weights}
      m = metrics_lib.make_ranking_metric_fn(metrics_lib.RankingMetricKey.DCG)
      m_w = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.DCG,
          weights_feature_name=weights_feature_name)
      expected_dcg_1 = _dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), expected_dcg_1),
      ])
      expected_dcg_2 = _dcg(2., 1) + _dcg(1., 2)
      expected_dcg_2_weighted = _dcg(2., 1) + _dcg(1., 2) * 2.
      expected_weight_2 = ((4 - 1) * 1. + (2 - 1) * 2.) / (4 - 1 + 2 - 1)
      self._check_metrics([
          (m(labels, scores,
             features), (expected_dcg_1 + expected_dcg_2) / 2.0),
          (m_w(labels, scores,
               features), (expected_dcg_1 + expected_dcg_2_weighted) /
           (1. + expected_weight_2)),
      ])
      # Test different gain and discount functions.
      gain_fn = lambda rel: rel
      rank_discount_fn = lambda rank: 1. / rank

      def mod_dcg_fn(l, r):
        return _dcg(l, r, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)

      m_mod = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.DCG,
          gain_fn=gain_fn,
          rank_discount_fn=rank_discount_fn)
      list_size = len(scores[0])
      expected_modified_dcg_1 = sum([
          mod_dcg_fn(labels[0][ind], ranks[0][ind]) for ind in range(list_size)
      ])
      self._check_metrics([
          (m_mod([labels[0]], [scores[0]], features), expected_modified_dcg_1),
      ])

  def test_ordered_pair_accuracy(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[-1., 0., 1.], [0., 1., 2.]]
      weights = [[1.], [2.]]
      item_weights = [[1., 1., 1.], [2., 2., 3.]]
      m = metrics_lib.ordered_pair_accuracy
      self._check_metrics([
          (m([labels[0]], [scores[0]]), 0.),
          (m([labels[1]], [scores[1]]), 1.),
          (m(labels, scores), (0. + 3.) / (1. + 3.)),
          (m(labels, scores, weights), (0. + 3. * 2.) / (1. + 3. * 2.)),
          (m(labels, scores,
             item_weights), (0. + 2. + 3. + 3.) / (1. + 2. + 3. + 3.)),
      ])

  def test_make_ordered_pair_accuracy_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      m = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ORDERED_PAIR_ACCURACY)
      self._check_metrics([
          (m([labels[0]], [scores[0]], {}), 1. / 2.),
          (m([labels[1]], [scores[1]], {}), 1.),
          (m(labels, scores, {}), (1. + 3.) / (2. + 3.)),
      ])

  def test_alpha_discounted_cumulative_gain(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      # ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[[0., 0.], [0., 1.], [0., 1.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      # cum_labels = [[[0., 2.], [0., 0.], [0., 1.]],
      #               [[2., 1.], [1., 1.], [0., 0.]]]
      m = metrics_lib.alpha_discounted_cumulative_gain
      expected_alphadcg = (_alpha_dcg([0., 1.], [0., 0.], 1) +
                           _alpha_dcg([0., 1.], [0., 1.], 2) +
                           _alpha_dcg([0., 0.], [0., 2.], 3))
      self._check_metrics([
          (m([labels[0]], [scores[0]]), expected_alphadcg),
      ])
      expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1) +
                             _alpha_dcg([0., 1.], [0., 1.], 2) +
                             _alpha_dcg([0., 0.], [0., 2.], 3))
      expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1) +
                             _alpha_dcg([1., 0.], [1., 1.], 2) +
                             _alpha_dcg([0., 0.], [2., 1.], 3))
      expected_alphadcg = (expected_alphadcg_1 + expected_alphadcg_2) / 2.0
      self._check_metrics([
          (m(labels, scores), expected_alphadcg),
      ])
      # Test different gain and discount functions.
      alpha = 0.2
      rank_discount_fn = lambda rank: 1. / rank

      mod_alpha_dcg_fn = functools.partial(_alpha_dcg, alpha=alpha,
                                           rank_discount_fn=rank_discount_fn)

      expected_modified_alphadcg_1 = (mod_alpha_dcg_fn([0., 1.], [0., 0.], 1) +
                                      mod_alpha_dcg_fn([0., 1.], [0., 1.], 2) +
                                      mod_alpha_dcg_fn([0., 0.], [0., 2.], 3))
      self._check_metrics([
          (m([labels[0]], [scores[0]],
             alpha=alpha,
             rank_discount_fn=rank_discount_fn), expected_modified_alphadcg_1),
      ])

  def test_alpha_discounted_cumulative_gain_with_zero_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      # cum_labels = [[[0., 0.], [0., 0.], [0., 0.]],
      #               [[2., 1.], [1., 1.], [0., 0.]]]

      expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1) +
                             _alpha_dcg([1., 0.], [1., 1.], 2) +
                             _alpha_dcg([0., 0.], [2., 1.], 3))
      m = metrics_lib.alpha_discounted_cumulative_gain
      self._check_metrics([(m(labels, scores),
                            (0. + expected_alphadcg_2) / 2.0)])

  def test_alpha_discounted_cumulative_gain_with_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 1.], [0., 1.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      # cum_labels = [[[0., 2.], [0., 0.], [0., 1.]],
      #               [[2., 1.], [1., 1.], [0., 0.]]]
      sum_labels = [[0., 1., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      list_weights = [[1.], [2.]]
      m = metrics_lib.alpha_discounted_cumulative_gain

      self._check_metrics([
          (m([labels[0]], [scores[0]], [weights[0]], topn=1),
           _alpha_dcg([0., 1.], [0., 0.], 1, 2.) / 2.5),
          (m([labels[0]], [scores[0]], [weights[0]]),
           (_alpha_dcg([0., 1.], [0., 0.], 1, 2.) +
            _alpha_dcg([0., 1.], [0., 1.], 2, 3.) +
            _alpha_dcg([0., 0.], [0., 2.], 3, 1.)) / 2.5),
      ])
      expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1, 2.) +
                             _alpha_dcg([0., 1.], [0., 1.], 2, 3.) +
                             _alpha_dcg([0., 0.], [0., 2.], 3, 1.)) / 2.5
      expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1, 6.) +
                             _alpha_dcg([1., 0.], [1., 1.], 2, 5.) +
                             _alpha_dcg([0., 0.], [2., 1.], 3, 4.)) / 5.5
      as_list_weights = _example_weights_to_list_weights(
          weights, sum_labels, 'ALPHADCG')
      expected_alphadcg = (
          expected_alphadcg_1 * as_list_weights[0] +
          expected_alphadcg_2 * as_list_weights[1]) / sum(as_list_weights)
      self._check_metrics([
          (m(labels, scores, weights), expected_alphadcg),
      ])
      expected_alphadcg_1 = _alpha_dcg([0., 1.], [0., 0.], 1, 2.) / 2.5
      expected_alphadcg_2 = _alpha_dcg([1., 1.], [0., 0.], 1, 6.) / 5.5
      expected_alphadcg = (
          expected_alphadcg_1 * as_list_weights[0] +
          expected_alphadcg_2 * as_list_weights[1]) / sum(as_list_weights)
      self._check_metrics([
          (m(labels, scores, weights, topn=1), expected_alphadcg),
      ])

      expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1, 1.) +
                             _alpha_dcg([0., 1.], [0., 1.], 2, 1.) +
                             _alpha_dcg([0., 0.], [0., 2.], 3, 1.))
      expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1, 2.) +
                             _alpha_dcg([1., 0.], [1., 1.], 2, 2.) +
                             _alpha_dcg([0., 0.], [2., 1.], 3, 2.)) / 2.
      expected_alphadcg = (expected_alphadcg_1 + 2. * expected_alphadcg_2) / 3.
      self._check_metrics([(m(labels, scores, list_weights),
                            expected_alphadcg)])
      # Test zero alphaDCG cases.
      self._check_metrics([
          (m(labels, scores, [[0.], [0.]]), 0.),
          (m([[[0., 0.], [0., 0.], [0., 0.]]],
             [scores[0]], weights[0], topn=1), 0.),
      ])

  def test_alpha_discounted_cumulative_gain_with_weights_zero_relevance(
      self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      # cum_labels = [[[0., 0.], [0., 0.], [0., 0.]],
      #               [[2., 1.], [1., 1.], [0., 0.]]]
      sum_labels = [[0., 0., 0.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      m = metrics_lib.alpha_discounted_cumulative_gain
      expected_alphadcg_1 = 0.0
      expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1, 6.) +
                             _alpha_dcg([1., 0.], [1., 1.], 2, 5.) +
                             _alpha_dcg([0., 0.], [2., 1.], 3, 4.)) / 5.5
      as_list_weights = _example_weights_to_list_weights(
          weights, sum_labels, 'ALPHADCG')
      self.assertAllClose(as_list_weights, [5.5, 5.5])
      expected_alphadcg = (
          expected_alphadcg_1 * as_list_weights[0] +
          expected_alphadcg_2 * as_list_weights[1]) / sum(as_list_weights)
      self._check_metrics([
          (m(labels, scores, weights), expected_alphadcg),
      ])
      # Test zero alphaDCG cases.
      self._check_metrics([(m(labels, scores, [[0.], [0.]]), 0.)])

  def test_make_alpha_discounted_cumulative_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      # Note that scores are ranked in descending order.
      # ranks = [[3, 1, 2], [3, 2, 1]]
      labels = [[[0., 0.], [0., 1.], [0., 1.]],
                [[0., 0.], [1., 0.], [1., 1.]]]
      # cum_labels = [[[0., 2.], [0., 0.], [0., 1.]],
      #               [[2., 1.], [1., 1.], [0., 0.]]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      weights_3d = [[[1.], [2.], [3.]], [[4.], [5.], [6.]]]
      list_weights = [1., 0.]
      list_weights_2d = [[1.], [0.]]
      weights_feature_name = 'weights'
      weights_invalid_feature_name = 'weights_invalid'
      weights_3d_feature_name = 'weights_3d'
      list_weights_name = 'list_weights'
      list_weights_2d_name = 'list_weights_2d'
      features = {
          weights_feature_name: [weights[0]],
          weights_invalid_feature_name: weights[0],
          weights_3d_feature_name: [weights_3d[0]],
          list_weights_name: list_weights,
          list_weights_2d_name: list_weights_2d
      }
      m = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG)

      expected_alphadcg = (_alpha_dcg([0., 1.], [0., 0.], 1) +
                           _alpha_dcg([0., 1.], [0., 1.], 2) +
                           _alpha_dcg([0., 0.], [0., 2.], 3))
      self._check_metrics([
          (m([labels[0]], [scores[0]], features), expected_alphadcg),
      ])
      expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1) +
                             _alpha_dcg([0., 1.], [0., 1.], 2) +
                             _alpha_dcg([0., 0.], [0., 2.], 3))
      expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1) +
                             _alpha_dcg([1., 0.], [1., 1.], 2) +
                             _alpha_dcg([0., 0.], [2., 1.], 3))
      expected_alphadcg = (expected_alphadcg_1 + expected_alphadcg_2) / 2.0
      self._check_metrics([
          (m(labels, scores, features), expected_alphadcg),
      ])

      # With item-wise weights.
      m_top = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG,
          weights_feature_name=weights_feature_name,
          topn=1)
      m_weight = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG,
          weights_feature_name=weights_feature_name)
      m_weights_3d = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG,
          weights_feature_name=weights_3d_feature_name)
      self._check_metrics([
          (m_top([labels[0]], [scores[0]], features),
           _alpha_dcg([0., 1.], [0., 0.], 1, 2.) / 2.5),
          (m_weight([labels[0]], [scores[0]], features),
           (_alpha_dcg([0., 1.], [0., 0.], 1, 2.) +
            _alpha_dcg([0., 1.], [0., 1.], 2, 3.) +
            _alpha_dcg([0., 0.], [0., 2.], 3, 1.)) / 2.5),
          (m_weights_3d([labels[0]], [scores[0]], features),
           (_alpha_dcg([0., 1.], [0., 0.], 1, 2.) +
            _alpha_dcg([0., 1.], [0., 1.], 2, 3.) +
            _alpha_dcg([0., 0.], [0., 2.], 3, 1.)) / 2.5),
      ])
      with self.assertRaises(ValueError):
        m_weight_invalid = metrics_lib.make_ranking_metric_fn(
            metrics_lib.RankingMetricKey.ALPHA_DCG,
            weights_feature_name=weights_invalid_feature_name)
        m_weight_invalid([labels[0]], [scores[0]], features)

      # With list-wise weights.
      m_list_weight = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG,
          weights_feature_name=list_weights_name)
      m_list_weight_2d = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG,
          weights_feature_name=list_weights_2d_name)
      self._check_metrics([
          (m_list_weight(labels, scores, features),
           (_alpha_dcg([0., 1.], [0., 0.], 1, 1.) +
            _alpha_dcg([0., 1.], [0., 1.], 2, 1.) +
            _alpha_dcg([0., 0.], [0., 2.], 3, 1.))),
          (m_list_weight_2d(labels, scores, features),
           (_alpha_dcg([0., 1.], [0., 0.], 1, 1.) +
            _alpha_dcg([0., 1.], [0., 1.], 2, 1.) +
            _alpha_dcg([0., 0.], [0., 2.], 3, 1.))),
      ])

      # Test different gain and discount functions.
      alpha = 0.2
      rank_discount_fn = lambda rank: 1. / rank

      mod_alpha_dcg_fn = functools.partial(_alpha_dcg, alpha=alpha,
                                           rank_discount_fn=rank_discount_fn)

      m_mod = metrics_lib.make_ranking_metric_fn(
          metrics_lib.RankingMetricKey.ALPHA_DCG,
          rank_discount_fn=rank_discount_fn,
          alpha=alpha)

      expected_modified_alphadcg_1 = (mod_alpha_dcg_fn([0., 1.], [0., 0.], 1) +
                                      mod_alpha_dcg_fn([0., 1.], [0., 1.], 2) +
                                      mod_alpha_dcg_fn([0., 0.], [0., 2.], 3))
      self._check_metrics([
          (m_mod([labels[0]], [scores[0]], features),
           expected_modified_alphadcg_1),
      ])

  def test_eval_metric(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.], [3., 1., 2.]]
      labels = [[0., 0., 1.], [0., 1., 2.], [0., 1., 0.]]
      weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
      gain_fn = lambda rel: rel
      rank_discount_fn = lambda rank: 1. / rank
      self._check_metrics([
          (metrics_lib.mean_reciprocal_rank(labels, scores),
           metrics_lib.eval_metric(
               metric_fn=metrics_lib.mean_reciprocal_rank,
               labels=labels,
               predictions=scores)),
          (metrics_lib.mean_reciprocal_rank(labels, scores, topn=1),
           metrics_lib.eval_metric(
               metric_fn=metrics_lib.mean_reciprocal_rank,
               labels=labels,
               predictions=scores,
               topn=1)),
          (metrics_lib.mean_reciprocal_rank(labels, scores, weights),
           metrics_lib.eval_metric(
               metric_fn=metrics_lib.mean_reciprocal_rank,
               labels=labels,
               predictions=scores,
               weights=weights)),
          (metrics_lib.discounted_cumulative_gain(
              labels,
              scores,
              gain_fn=gain_fn,
              rank_discount_fn=rank_discount_fn),
           metrics_lib.eval_metric(
               metric_fn=metrics_lib.discounted_cumulative_gain,
               labels=labels,
               predictions=scores,
               gain_fn=gain_fn,
               rank_discount_fn=rank_discount_fn)),
      ])

  def test_compute_mean(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      with self.test_session() as sess:
        for key in [
            'mrr',
            'arp',
            'ndcg',
            'dcg',
            'precision',
            'map',
            'ordered_pair_accuracy',
        ]:
          value = sess.run(
              metrics_lib.compute_mean(
                  key, labels, scores, weights, 2, name=key))
          self.assertGreater(value, 0.)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
