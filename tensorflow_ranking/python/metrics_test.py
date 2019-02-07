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

"""Tests for ranking metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow_ranking.python import metrics


def _dcg(label, rank, weight=1.0):
  """Returns a single dcg addend.

  Args:
    label: The document label.
    rank: The document rank starting from 1.
    weight: The document weight.

  Returns:
    A single dcg addend. e.g. weight*(2^relevance-1)/log2(rank+1).
  """
  return weight * (math.pow(2.0, label) - 1.0) / math.log(rank + 1.0, 2.0)


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
  for example_weights, labels in zip(weights, relevances):
    boosted_labels = [_label_boost(boost_form, label) for label in labels]
    numerator = sum((weight * boosted_labels[i])
                    for i, weight in enumerate(example_weights))
    denominator = sum(boosted_labels)
    list_weights.append(0.0 if denominator == 0.0 else numerator / denominator)
  return list_weights


class MetricsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()
    ops.reset_default_graph()

  def _check_metrics(self, metrics_and_values):
    """Checks metrics against values."""
    with self.test_session() as sess:
      sess.run(variables.local_variables_initializer())
      for (metric_op, update_op), value in metrics_and_values:
        sess.run(update_op)
        self.assertAlmostEqual(sess.run(metric_op), value, places=5)

  def test_reset_invalid_labels(self):
    scores = [[1., 3., 2.]]
    labels = [[0., -1., 1.]]
    labels, predictions, _, _ = metrics._prepare_and_validate_params(
        labels, scores)
    self.assertAllClose(labels, [[0., 0., 1.]])
    self.assertAllClose(predictions, [[1., 1. - 1e-6, 2]])

  def test_mean_reciprocal_rank(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    m = metrics.mean_reciprocal_rank
    self._check_metrics([
        (m([labels[0]], [scores[0]]), 0.5),
        (m(labels, scores), (0.5 + 1.0) / 2),
        (m(labels, scores, weights), (6. * 0.5 + 15. * 1.) / (6. + 15.)),
    ])

  def test_make_mean_reciprocal_rank_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    m = metrics.make_ranking_metric_fn(metrics.RankingMetricKey.MRR)
    m_w = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.MRR, weights_feature_name=weights_feature_name)
    self._check_metrics([
        (m([labels[0]], [scores[0]], features), 0.5),
        (m(labels, scores, features), (0.5 + 1.0) / 2),
        (m_w(labels, scores, features), (6. * 0.5 + 15. * 1.) / (6. + 15.)),
    ])

  def test_average_relevance_position(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    m = metrics.average_relevance_position
    self._check_metrics([
        (m([labels[0]], [scores[0]]), 2.),
        (m(labels, scores), (1. * 2. + 2. * 1. + 1. * 2.) / 4.),
        (m(labels, scores, weights),
         (3. * 1. * 2. + 6. * 2. * 1. + 5 * 1. * 2.) / (3. + 12. + 5.)),
    ])

  def test_make_average_relevance_position_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    m = metrics.make_ranking_metric_fn(metrics.RankingMetricKey.ARP)
    m_w = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.ARP, weights_feature_name=weights_feature_name)
    self._check_metrics([
        (m([labels[0]], [scores[0]], features), 2.),
        (m(labels, scores, features), (1. * 2. + 2. * 1. + 1. * 2.) / 4.),
        (m_w(labels, scores, features),
         (3. * 1. * 2. + 6. * 2. * 1. + 5 * 1. * 2.) / (3. + 12. + 5.)),
    ])

  def test_precision(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    m = metrics.precision
    self._check_metrics([
        (m([labels[0]], [scores[0]]), 1. / 3.),
        (m([labels[0]], [scores[0]], topn=1), 0. / 1.),
        (m([labels[0]], [scores[0]], topn=2), 1. / 2.),
        (m(labels, scores), (1. / 3. + 2. / 3.) / 2.),
    ])

  def test_precision_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]
    m = metrics.precision
    as_list_weights = _example_weights_to_list_weights(weights, labels,
                                                       'PRECISION')
    self._check_metrics([
        (m(labels, scores, weights),
         ((3. / 6.) * as_list_weights[0] +
          (11. / 15.) * as_list_weights[1]) / sum(as_list_weights)),
        (m(labels, scores, weights, topn=2),
         ((3. / 5.) * as_list_weights[0] +
          (11. / 11.) * as_list_weights[1]) / sum(as_list_weights)),
        # Per list weight.
        (m(labels, scores, list_weights),
         ((1. / 3.) * list_weights[0][0] + (4. / 6.) * list_weights[1][0]) / 3.0
        ),
        # Zero precision case.
        (m(labels, scores, [0., 0., 0.], topn=2), 0.),
    ])

  def test_make_precision_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    features = {}
    m = metrics.make_ranking_metric_fn(metrics.RankingMetricKey.PRECISION)
    m_top_1 = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.PRECISION, topn=1)
    m_top_2 = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.PRECISION, topn=2)
    self._check_metrics([
        (m([labels[0]], [scores[0]], features), 1. / 3.),
        (m_top_1([labels[0]], [scores[0]], features), 0. / 1.),
        (m_top_2([labels[0]], [scores[0]], features), 1. / 2.),
        (m(labels, scores, features), (1. / 3. + 2. / 3.) / 2.),
    ])

  def test_normalized_discounted_cumulative_gain(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    m = metrics.normalized_discounted_cumulative_gain
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

  def test_normalized_discounted_cumulative_gain_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]
    m = metrics.normalized_discounted_cumulative_gain
    self._check_metrics([
        (m([labels[0]], [scores[0]], weights[0], topn=1),
         _dcg(0., 1, 2.) / _dcg(1., 1, 3.)),
        (m([labels[0]], [scores[0]],
           weights[0]), (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
         (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
    ])
    expected_ndcg_1 = (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) / (
        _dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))
    expected_ndcg_2 = 1.0
    as_list_weights = _example_weights_to_list_weights(weights, labels, 'NDCG')
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

  def test_normalized_discounted_cumulative_gain_with_zero_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    m = metrics.normalized_discounted_cumulative_gain
    self._check_metrics([
        (m(labels, scores, [[0.], [0.]]), 0.),
        (m([[0., 0., 0.]], [scores[0]], weights[0], topn=1), 0.),
    ])

  def test_make_normalized_discounted_cumulative_gain_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights[0]}
    m = metrics.make_ranking_metric_fn(metrics.RankingMetricKey.NDCG)

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

    # With weights.
    m_top = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.NDCG,
        weights_feature_name=weights_feature_name,
        topn=1)
    m_weight = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.NDCG,
        weights_feature_name=weights_feature_name)
    self._check_metrics([
        (m_top([labels[0]], [scores[0]], features),
         _dcg(0., 1, 2.) / _dcg(1., 1, 3.)),
        (m_weight([labels[0]], [scores[0]], features),
         (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) /
         (_dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))),
    ])

  def test_discounted_cumulative_gain(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 1., 1.], [2., 2., 1.]]
    m = metrics.discounted_cumulative_gain
    expected_dcg_1 = _dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)
    self._check_metrics([
        (m([labels[0]], [scores[0]]), expected_dcg_1),
    ])
    expected_dcg_2 = _dcg(2., 1) + _dcg(1., 2)
    expected_dcg_2_weighted = _dcg(2., 1) + _dcg(1., 2) * 2.
    expected_weight_2 = ((4 - 1) * 1. + (2 - 1) * 2.) / (4 - 1 + 2 - 1)
    self._check_metrics([
        (m(labels, scores), (expected_dcg_1 + expected_dcg_2) / 2.0),
        (m(labels, scores, weights),
         (expected_dcg_1 + expected_dcg_2_weighted) / (1. + expected_weight_2)),
    ])

  def test_make_discounted_cumulative_gain_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 1., 1.], [2., 2., 1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: weights}
    m = metrics.make_ranking_metric_fn(metrics.RankingMetricKey.DCG)
    m_w = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.DCG, weights_feature_name=weights_feature_name)
    expected_dcg_1 = _dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)
    self._check_metrics([
        (m([labels[0]], [scores[0]], features), expected_dcg_1),
    ])
    expected_dcg_2 = _dcg(2., 1) + _dcg(1., 2)
    expected_dcg_2_weighted = _dcg(2., 1) + _dcg(1., 2) * 2.
    expected_weight_2 = ((4 - 1) * 1. + (2 - 1) * 2.) / (4 - 1 + 2 - 1)
    self._check_metrics([
        (m(labels, scores, features), (expected_dcg_1 + expected_dcg_2) / 2.0),
        (m_w(labels, scores, features),
         (expected_dcg_1 + expected_dcg_2_weighted) / (1. + expected_weight_2)),
    ])

  def test_ordered_pair_accuracy(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[-1., 0., 1.], [0., 1., 2.]]
    weights = [[1.], [2.]]
    item_weights = [[1., 1., 1.], [2., 2., 3.]]
    m = metrics.ordered_pair_accuracy
    self._check_metrics([
        (m([labels[0]], [scores[0]]), 0.),
        (m([labels[1]], [scores[1]]), 1.),
        (m(labels, scores), (0. + 3.) / (1. + 3.)),
        (m(labels, scores, weights), (0. + 3. * 2.) / (1. + 3. * 2.)),
        (m(labels, scores, item_weights),
         (0. + 2. + 3. + 3.) / (1. + 2. + 3. + 3.)),
    ])

  def test_make_ordered_pair_accuracy_fn(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    m = metrics.make_ranking_metric_fn(
        metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY)
    self._check_metrics([
        (m([labels[0]], [scores[0]], {}), 1. / 2.),
        (m([labels[1]], [scores[1]], {}), 1.),
        (m(labels, scores, {}), (1. + 3.) / (2. + 3.)),
    ])


if __name__ == '__main__':
  test.main()
