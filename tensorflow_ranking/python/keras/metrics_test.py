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

"""Tests for Keras ranking metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import six
import tensorflow.compat.v2 as tf

from tensorflow_ranking.python.keras import metrics as metrics_lib


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


class MetricsSerializationTest(tf.test.TestCase):

  def _check_config(self, metric_cls, init_args):
    """Checks if Keras metrics can be saved and restored.

    Args:
      metric_cls: Keras metric class.
      init_args: (dict) Initializer keyword arguments.
    """
    init_args.update({
        'name': 'my_metric',
        'dtype': tf.float32,
    })
    metric_obj = metric_cls(**init_args)
    config = metric_obj.get_config()
    self.assertIsNotNone(config)

    restored_metric_obj = metric_cls.from_config(config)
    for init_name, init_value in six.iteritems(init_args):
      self.assertEqual(init_value, getattr(restored_metric_obj,
                                           '_' + init_name))

  def test_mean_reciprocal_rank(self):
    self._check_config(metrics_lib.MRRMetric, {'topn': 1})

  def test_average_relevance_position(self):
    self._check_config(metrics_lib.ARPMetric, {})

  def test_precision(self):
    self._check_config(metrics_lib.PrecisionMetric, {'topn': 1})

  def test_precision_ia(self):
    self._check_config(metrics_lib.PrecisionIAMetric, {'topn': 1})

  def test_mean_average_precision(self):
    self._check_config(metrics_lib.MeanAveragePrecisionMetric, {'topn': 1})

  def test_normalized_discounted_cumulative_gain(self):
    self._check_config(
        metrics_lib.NDCGMetric, {
            'topn': 1,
            'gain_fn': lambda rel: rel,
            'rank_discount_fn': lambda rank: rank
        })

  def test_discounted_cumulative_gain(self):
    gain_fn = lambda rel: rel
    rank_discount_fn = lambda rank: rank
    self._check_config(metrics_lib.DCGMetric, {
        'topn': 1,
        'gain_fn': gain_fn,
        'rank_discount_fn': rank_discount_fn
    })

  def test_alpha_discounted_cumulative_gain(self):
    rank_discount_fn = lambda rank: rank
    self._check_config(metrics_lib.AlphaDCGMetric, {
        'topn': 1,
        'alpha': 0.5,
        'rank_discount_fn': rank_discount_fn,
        'seed': 1,
    })

  def test_ordered_pair_accuracy(self):
    self._check_config(metrics_lib.OPAMetric, {})


class MetricsTest(tf.test.TestCase):

  def test_mean_reciprocal_rank(self):
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

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 1. / rel_rank[0]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=1)
    metric_.update_state([labels[0]], [scores[0]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

    metric_ = metrics_lib.MRRMetric(topn=2)
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 1. / rel_rank[0]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state([labels[1]], [scores[1]])
    expected_result = 1. / rel_rank[1]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=1)
    metric_.update_state([labels[1]], [scores[1]])
    expected_result = 1. / rel_rank[1]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=6)
    metric_.update_state([labels[1]], [scores[1]])
    expected_result = 1. / rel_rank[1]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state([labels[2]], [scores[2]])
    expected_result = 1. / rel_rank[2]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=1)
    metric_.update_state([labels[2]], [scores[2]])
    self.assertAlmostEqual(metric_.result().numpy(), 0, places=5)

    metric_ = metrics_lib.MRRMetric(topn=2)
    metric_.update_state([labels[2]], [scores[2]])
    self.assertAlmostEqual(metric_.result().numpy(), 0, places=5)

    metric_ = metrics_lib.MRRMetric(topn=3)
    metric_.update_state([labels[2]], [scores[2]])
    expected_result = 1. / rel_rank[2]
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state(labels[:2], scores[:2])
    expected_result = (0.5 + 1.0) / 2
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state(labels[:2], scores[:2], weights[:2])
    expected_result = (3. * 0.5 + (6. + 5.) / 2. * 1.) / (3. + (6. + 5) / 2.)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state(labels, scores)
    expected_result = sum([1. / rel_rank[ind] for ind in range(num_queries)
                          ]) / num_queries
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=1)
    metric_.update_state(labels, scores)
    expected_result = sum([0., 1. / rel_rank[1], 0.]) / num_queries
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=2)
    metric_.update_state(labels, scores)
    expected_result = sum([1. / rel_rank[0], 1. / rel_rank[1], 0.
                          ]) / num_queries
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric()
    metric_.update_state(labels, scores, weights)
    expected_result = sum([
        mean_relevant_weights[ind] / rel_rank[ind] for ind in range(num_queries)
    ]) / sum(mean_relevant_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MRRMetric(topn=1)
    metric_.update_state(labels, scores, weights)
    expected_result = sum([0., mean_relevant_weights[1] / rel_rank[1], 0.
                          ]) / sum(mean_relevant_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_average_relevance_position(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]

    metric_ = metrics_lib.ARPMetric()
    metric_.update_state([labels[0]], [scores[0]])
    self.assertAlmostEqual(metric_.result().numpy(), 2., places=5)

    metric_ = metrics_lib.ARPMetric()
    metric_.update_state(labels, scores)
    expected_result = (1. * 2. + 2. * 1. + 1. * 2.) / 4.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.ARPMetric()
    metric_.update_state(labels, scores, weights)
    expected_result = (3. * 1. * 2. + 6. * 2. * 1. + 5 * 1. * 2.) / (3. + 12. +
                                                                     5.)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_precision(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]

    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 1. / 3.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionMetric(topn=1)
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 0. / 1.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state(labels, scores)
    expected_result = (1. / 3. + 2. / 3.) / 2.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_precision_with_zero_relevance(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 0.], [0., 1., 2.]]

    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 0.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state(labels, scores)
    expected_result = (0. + 2. / 3.) / 2.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_precision_weights_with_zero_relevance(self):
    scores = [[1., 3., 2.], [1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 0.], [0., 0., 1.], [0., 1., 2.]]
    weights = [[0., 0., 1.], [1., 2., 3.], [4., 5., 6.]]
    as_list_weights = _example_weights_to_list_weights(weights, labels,
                                                       'PRECISION')
    self.assertAllClose(as_list_weights, [(3 + 5.5) / 2., 3, 5.5])
    metric_ = metrics_lib.PrecisionMetric(topn=2)
    metric_.update_state(labels, scores, weights)
    expected_result = (0.0 * as_list_weights[0] +
                       (1. / 2.) * as_list_weights[1] +
                       (2. / 2.) * as_list_weights[2]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state(labels[0:2], scores[0:2], [[0., 0., 0.], [0., 0., 0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_precision_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]
    as_list_weights = _example_weights_to_list_weights(weights, labels,
                                                       'PRECISION')

    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state(labels, scores, weights)
    expected_result = ((1. / 3.) * as_list_weights[0] +
                       (2. / 3.) * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionMetric(topn=2)
    metric_.update_state(labels, scores, weights)
    expected_result = ((1. / 2.) * as_list_weights[0] +
                       (2. / 2.) * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    # Per list weight.
    metric_ = metrics_lib.PrecisionMetric()
    metric_.update_state(labels, scores, list_weights)
    expected_result = ((1. / 3.) * list_weights[0][0] +
                       (2. / 3.) * list_weights[1][0]) / 3.0
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    # Zero precision case.
    metric_ = metrics_lib.PrecisionMetric(topn=2)
    metric_.update_state(labels, scores, [[0., 0., 0.], [0., 0., 0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_precision_ia(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 0.], [1., 0.]],
              [[0., 0.], [1., 0.], [1., 1.]]]

    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 1. / 3.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionIAMetric(topn=1)
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 0. / 1.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state(labels, scores)
    expected_result = (1. / 3. + 3. / 6.) / 2.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_precision_ia_with_zero_relevance(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 0.], [0., 0.]],
              [[0., 0.], [1., 0.], [1., 1.]]]

    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = 0.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state(labels, scores)
    expected_result = (0. + 3. / 6.) / 2.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_precision_ia_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 0.], [1., 0.]],
              [[0., 0.], [1., 0.], [1., 1.]]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]
    as_list_weights = _example_weights_to_list_weights(
        weights, [[0., 0., 1.], [0., 1., 1.]], 'PRECISION')

    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state(labels, scores, weights)
    expected_result = ((1. / 3.) * as_list_weights[0] +
                       (3. / 6.) * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionIAMetric(topn=2)
    metric_.update_state(labels, scores, weights)
    expected_result = ((1. / 2.) * as_list_weights[0] +
                       (3. / 4.) * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    # Per list weight.
    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state(labels, scores, list_weights)
    expected_result = ((1. / 3.) * list_weights[0][0] +
                       (3. / 6.) * list_weights[1][0]) / 3.0
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    # Zero precision case.
    metric_ = metrics_lib.PrecisionIAMetric(topn=2)
    metric_.update_state(labels, scores, [[0., 0., 0.], [0., 0., 0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_precision_ia_weights_with_zero_relevance(self):
    scores = [[1., 3., 2.], [1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 0.], [0., 0.]],
              [[0., 0.], [0., 0.], [1., 0.]],
              [[0., 0.], [1., 0.], [1., 1.]]]
    weights = [[0., 0., 1.], [1., 2., 3.], [4., 5., 6.]]
    as_list_weights = _example_weights_to_list_weights(
        weights, [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.]], 'PRECISION')
    self.assertAllClose(as_list_weights, [(3 + 5.5) / 2., 3, 5.5])
    metric_ = metrics_lib.PrecisionIAMetric(topn=2)
    metric_.update_state(labels, scores, weights)
    expected_result = (0.0 * as_list_weights[0] +
                       (1. / 2.) * as_list_weights[1] +
                       (3. / 4.) * as_list_weights[2]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.PrecisionIAMetric()
    metric_.update_state(labels[0:2], scores[0:2], [[0., 0., 0.], [0., 0., 0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_mean_average_precision(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    # Note that scores are ranked in descending order, so the ranks are
    # [[3, 1, 2], [3, 2, 1]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    rels = [[0, 0, 1], [0, 1, 1]]

    metric_ = metrics_lib.MeanAveragePrecisionMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = _ap(rels[0], scores[0])
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric(topn=1)
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = _ap(rels[0], scores[0], topn=1)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric(topn=2)
    metric_.update_state([labels[0]], [scores[0]])
    expected_result = _ap(rels[0], scores[0], topn=2)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric()
    metric_.update_state(labels, scores)
    expected_result = sum(_ap(rels[i], scores[i]) for i in range(2)) / 2.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric(topn=1)
    metric_.update_state(labels, scores)
    expected_result = sum(
        _ap(rels[i], scores[i], topn=1) for i in range(2)) / 2.
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_mean_average_precision_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    # Note that scores are ranked in descending order, so the ranks are
    # [[3, 1, 2], [3, 2, 1]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    rels = [[0, 0, 1], [0, 1, 1]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]
    as_list_weights = _example_weights_to_list_weights(weights, labels, 'MAP')
    # See Equation (1.7) in the following reference to make sense of
    # the formulas that appear in the following expression:
    # Liu, T-Y "Learning to Rank for Information Retrieval" found at
    # https://www.nowpublishers.com/article/DownloadSummary/INR-016

    metric_ = metrics_lib.MeanAveragePrecisionMetric()
    metric_.update_state([labels[0]], [scores[0]], [weights[0]])
    expected_result = ((1. / 2.) * 3.) / (0 * 1 + 0 * 2 + 1 * 3)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric()
    metric_.update_state([labels[1]], [scores[1]], [weights[1]])
    expected_result = ((1. / 1.) * 6. +
                       (2. / 2.) * 5.) / (0 * 4 + 1 * 5 + 1 * 6)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric()
    metric_.update_state(labels, scores, weights)
    expected_result = (
        ((1. / 2.) * 3.) / (0 * 1 + 0 * 2 + 1 * 3) * as_list_weights[0] +
        ((1. / 1.) * 6. + (2. / 2.) * 5.) /
        (0 * 4 + 1 * 5 + 1 * 6) * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric(topn=1)
    metric_.update_state(labels, scores, weights)
    expected_result = ((0 * as_list_weights[0] + ((1. / 1.) * 6.) /
                        (1 * 6) * as_list_weights[1]) / sum(as_list_weights))
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.MeanAveragePrecisionMetric(topn=2)
    metric_.update_state(labels, scores, weights)
    expected_result = (
        ((1. / 2.) * 3.) / (0 * 1 + 1 * 3) * as_list_weights[0] +
        ((1. / 1.) * 6. + (2. / 2.) * 5.) /
        (1 * 5 + 1 * 6) * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    # Per list weight.
    metric_ = metrics_lib.MeanAveragePrecisionMetric()
    metric_.update_state(labels, scores, list_weights)
    expected_result = sum(
        _ap(rels[i], scores[i]) * list_weights[i][0] for i in range(2)) / sum(
            list_weights[i][0] for i in range(2))
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    # Zero precision case.
    metric_ = metrics_lib.MeanAveragePrecisionMetric(topn=2)
    metric_.update_state(labels, scores, [0., 0., 0.])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_normalized_discounted_cumulative_gain(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    # Note that scores are ranked in descending order.
    ranks = [[3, 1, 2], [3, 2, 1]]
    labels = [[0., 0., 1.], [0., 1., 2.]]

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_ndcg = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
        _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
    self.assertAlmostEqual(metric_.result().numpy(), expected_ndcg, places=5)

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores)
    expected_ndcg_1 = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
        _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
    expected_ndcg_2 = 1.0
    expected_ndcg = (expected_ndcg_1 + expected_ndcg_2) / 2.0
    self.assertAlmostEqual(metric_.result().numpy(), expected_ndcg, places=5)

    # Test different gain and discount functions.
    gain_fn = lambda rel: rel
    rank_discount_fn = lambda rank: rank
    metric_ = metrics_lib.NDCGMetric(
        gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)
    metric_.update_state([labels[0]], [scores[0]])

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
    self.assertAlmostEqual(
        metric_.result().numpy(), expected_modified_ndcg_1, places=5)

  def test_normalized_discounted_cumulative_gain_with_zero_relevance(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 0.], [0., 1., 2.]]

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores)
    self.assertAlmostEqual(metric_.result().numpy(), (0. + 1.) / 2.0, places=5)

  def test_normalized_discounted_cumulative_gain_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]

    metric_ = metrics_lib.NDCGMetric(topn=1)
    metric_.update_state([labels[0]], [scores[0]], weights[0])
    expected_result = _dcg(0., 1, 2.) / _dcg(1., 1, 3.)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state([labels[0]], [scores[0]], weights[0])
    expected_result = (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) / (
        _dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores, weights)
    expected_ndcg_1 = (_dcg(0., 1, 2.) + _dcg(1., 2, 3.) + _dcg(0., 3, 1.)) / (
        _dcg(1., 1, 3.) + _dcg(0., 2, 1.) + _dcg(0., 3, 2.))
    expected_ndcg_2 = 1.0
    as_list_weights = _example_weights_to_list_weights(weights, labels, 'NDCG')
    expected_ndcg = (expected_ndcg_1 * as_list_weights[0] + expected_ndcg_2 *
                     as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_ndcg, places=5)

    metric_ = metrics_lib.NDCGMetric(topn=1)
    metric_.update_state(labels, scores, weights)
    expected_ndcg_1 = _dcg(0., 1, 2.) / _dcg(1., 1, 3.)
    expected_ndcg_2 = 1.0
    expected_ndcg = (expected_ndcg_1 * as_list_weights[0] + expected_ndcg_2 *
                     as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_ndcg, places=5)

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores, list_weights)
    expected_ndcg_1 = (_dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)) / (
        _dcg(1., 1) + _dcg(0., 2) + _dcg(0., 3))
    expected_ndcg_2 = 1.0
    expected_ndcg = (expected_ndcg_1 + 2. * expected_ndcg_2) / 3.0
    self.assertAlmostEqual(metric_.result().numpy(), expected_ndcg, places=5)

    # Test zero NDCG cases.
    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores, [[0.], [0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

    metric_ = metrics_lib.NDCGMetric(topn=1)
    metric_.update_state([[0., 0., 0.]], [scores[0]], weights[0])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_normalized_discounted_cumulative_gain_with_weights_zero_relevance(
      self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 0.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    expected_ndcg_1 = 0.0
    expected_ndcg_2 = 1.0
    as_list_weights = _example_weights_to_list_weights(weights, labels, 'NDCG')
    self.assertAllClose(as_list_weights, [5.75, 5.75])
    expected_ndcg = (expected_ndcg_1 * as_list_weights[0] + expected_ndcg_2 *
                     as_list_weights[1]) / sum(as_list_weights)
    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores, weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_ndcg, places=5)

    # Test zero NDCG cases.
    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores, [[0.], [0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_normalized_discounted_cumulative_gain_with_zero_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]

    metric_ = metrics_lib.NDCGMetric()
    metric_.update_state(labels, scores, [[0.], [0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

    metric_ = metrics_lib.NDCGMetric(topn=1)
    metric_.update_state([[0., 0., 0.]], [scores[0]], weights[0])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_discounted_cumulative_gain(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    # Note that scores are ranked in descending order.
    ranks = [[3, 1, 2], [3, 2, 1]]
    labels = [[0., 0., 1.], [0., 1., 2.]]
    weights = [[1., 1., 1.], [2., 2., 1.]]

    expected_dcg_1 = _dcg(0., 1) + _dcg(1., 2) + _dcg(0., 3)
    expected_dcg_2 = _dcg(2., 1) + _dcg(1., 2)
    expected_dcg_2_weighted = _dcg(2., 1) + _dcg(1., 2) * 2.
    expected_weight_2 = ((4 - 1) * 1. + (2 - 1) * 2.) / (4 - 1 + 2 - 1)

    metric_ = metrics_lib.DCGMetric()
    metric_.update_state([labels[0]], [scores[0]])
    self.assertAlmostEqual(metric_.result().numpy(), expected_dcg_1, places=5)

    metric_ = metrics_lib.DCGMetric()
    metric_.update_state(labels, scores)
    expected_dcg = (expected_dcg_1 + expected_dcg_2) / 2.0
    self.assertAlmostEqual(metric_.result().numpy(), expected_dcg, places=5)

    metric_ = metrics_lib.DCGMetric()
    metric_.update_state(labels, scores, weights)
    expected_dcg = (expected_dcg_1 +
                    expected_dcg_2_weighted) / (1. + expected_weight_2)
    self.assertAlmostEqual(metric_.result().numpy(), expected_dcg, places=5)

    # Test different gain and discount functions.
    gain_fn = lambda rel: rel
    rank_discount_fn = lambda rank: 1. / rank

    def mod_dcg_fn(l, r):
      return _dcg(l, r, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)

    list_size = len(scores[0])
    expected_modified_dcg_1 = sum(
        [mod_dcg_fn(labels[0][ind], ranks[0][ind]) for ind in range(list_size)])

    metric_ = metrics_lib.DCGMetric(
        gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)
    metric_.update_state([labels[0]], [scores[0]])
    self.assertAlmostEqual(
        metric_.result().numpy(), expected_modified_dcg_1, places=5)

  def test_alpha_discounted_cumulative_gain(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    # Note that scores are ranked in descending order.
    # ranks = [[3, 1, 2], [3, 2, 1]]
    labels = [[[0., 0.], [0., 1.], [0., 1.]],
              [[0., 0.], [1., 0.], [1., 1.]]]
    # cum_labels = [[[0., 2.], [0., 0.], [0., 1.]],
    #               [[2., 1.], [1., 1.], [0., 0.]]]

    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state([labels[0]], [scores[0]])
    expected_alphadcg = (_alpha_dcg([0., 1.], [0., 0.], 1) +
                         _alpha_dcg([0., 1.], [0., 1.], 2) +
                         _alpha_dcg([0., 0.], [0., 2.], 3))
    self.assertAlmostEqual(metric_.result().numpy(), expected_alphadcg,
                           places=5)

    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores)
    expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1) +
                           _alpha_dcg([0., 1.], [0., 1.], 2) +
                           _alpha_dcg([0., 0.], [0., 2.], 3))
    expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1) +
                           _alpha_dcg([1., 0.], [1., 1.], 2) +
                           _alpha_dcg([0., 0.], [2., 1.], 3))
    expected_alphadcg = (expected_alphadcg_1 + expected_alphadcg_2) / 2.0
    self.assertAlmostEqual(metric_.result().numpy(), expected_alphadcg,
                           places=5)

   # Test different gain and discount functions.
    alpha = 0.2
    rank_discount_fn = lambda rank: 1. / rank

    mod_alpha_dcg_fn = functools.partial(_alpha_dcg, alpha=alpha,
                                         rank_discount_fn=rank_discount_fn)

    metric_ = metrics_lib.AlphaDCGMetric(
        alpha=alpha, rank_discount_fn=rank_discount_fn)
    metric_.update_state([labels[0]], [scores[0]])
    expected_modified_alphadcg_1 = (mod_alpha_dcg_fn([0., 1.], [0., 0.], 1) +
                                    mod_alpha_dcg_fn([0., 1.], [0., 1.], 2) +
                                    mod_alpha_dcg_fn([0., 0.], [0., 2.], 3))
    self.assertAlmostEqual(
        metric_.result().numpy(), expected_modified_alphadcg_1, places=5)

  def test_alpha_discounted_cumulative_gain_with_zero_relevance(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 0.], [0., 0.]],
              [[0., 0.], [1., 0.], [1., 1.]]]
    # cum_labels = [[[0., 0.], [0., 0.], [0., 0.]],
    #               [[2., 1.], [1., 1.], [0., 0.]]]

    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores)
    expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1) +
                           _alpha_dcg([1., 0.], [1., 1.], 2) +
                           _alpha_dcg([0., 0.], [2., 1.], 3))
    self.assertAlmostEqual(metric_.result().numpy(),
                           (0. + expected_alphadcg_2) / 2.0, places=5)

  def test_alpha_discounted_cumulative_gain_with_weights(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 1.], [0., 1.]],
              [[0., 0.], [1., 0.], [1., 1.]]]
    # cum_labels = [[[0., 2.], [0., 0.], [0., 1.]],
    #               [[2., 1.], [1., 1.], [0., 0.]]]
    sum_labels = [[0., 1., 1.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    list_weights = [[1.], [2.]]

    metric_ = metrics_lib.AlphaDCGMetric(topn=1)
    metric_.update_state([labels[0]], [scores[0]], weights[0])
    expected_result = _alpha_dcg([0., 1.], [0., 0.], 1, 2.) / 2.5
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state([labels[0]], [scores[0]], weights[0])
    expected_result = (_alpha_dcg([0., 1.], [0., 0.], 1, 2.) +
                       _alpha_dcg([0., 1.], [0., 1.], 2, 3.) +
                       _alpha_dcg([0., 0.], [0., 2.], 3, 1.)) / 2.5
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores, weights)
    expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1, 2.) +
                           _alpha_dcg([0., 1.], [0., 1.], 2, 3.) +
                           _alpha_dcg([0., 0.], [0., 2.], 3, 1.)) / 2.5
    expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1, 6.) +
                           _alpha_dcg([1., 0.], [1., 1.], 2, 5.) +
                           _alpha_dcg([0., 0.], [2., 1.], 3, 4.)) / 5.5
    as_list_weights = _example_weights_to_list_weights(weights, sum_labels,
                                                       'ALPHADCG')
    expected_alphadcg = (
        expected_alphadcg_1 * as_list_weights[0] +
        expected_alphadcg_2 * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_alphadcg,
                           places=5)

    metric_ = metrics_lib.AlphaDCGMetric(topn=1)
    metric_.update_state(labels, scores, weights)
    expected_alphadcg_1 = _alpha_dcg([0., 1.], [0., 0.], 1, 2.) / 2.5
    expected_alphadcg_2 = _alpha_dcg([1., 1.], [0., 0.], 1, 6.) / 5.5
    expected_alphadcg = (
        expected_alphadcg_1 * as_list_weights[0] +
        expected_alphadcg_2 * as_list_weights[1]) / sum(as_list_weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_alphadcg,
                           places=5)

    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores, list_weights)
    expected_alphadcg_1 = (_alpha_dcg([0., 1.], [0., 0.], 1, 1.) +
                           _alpha_dcg([0., 1.], [0., 1.], 2, 1.) +
                           _alpha_dcg([0., 0.], [0., 2.], 3, 1.))
    expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1, 2.) +
                           _alpha_dcg([1., 0.], [1., 1.], 2, 2.) +
                           _alpha_dcg([0., 0.], [2., 1.], 3, 2.)) / 2.
    expected_alphadcg = (expected_alphadcg_1 + 2. * expected_alphadcg_2) / 3.
    self.assertAlmostEqual(metric_.result().numpy(), expected_alphadcg,
                           places=5)

    # Test zero alphaDCG cases.
    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores, [[0.], [0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

    metric_ = metrics_lib.AlphaDCGMetric(topn=1)
    metric_.update_state([[[0., 0.], [0., 0.], [0., 0.]]], [scores[0]],
                         weights[0])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_alpha_discounted_cumulative_gain_with_weights_zero_relevance(
      self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[[0., 0.], [0., 0.], [0., 0.]],
              [[0., 0.], [1., 0.], [1., 1.]]]
    # cum_labels = [[[0., 0.], [0., 0.], [0., 0.]],
    #               [[2., 1.], [1., 1.], [0., 0.]]]
    sum_labels = [[0., 0., 0.], [0., 1., 2.]]
    weights = [[1., 2., 3.], [4., 5., 6.]]
    expected_alphadcg_1 = 0.0
    expected_alphadcg_2 = (_alpha_dcg([1., 1.], [0., 0.], 1, 6.) +
                           _alpha_dcg([1., 0.], [1., 1.], 2, 5.) +
                           _alpha_dcg([0., 0.], [2., 1.], 3, 4.)) / 5.5
    as_list_weights = _example_weights_to_list_weights(weights, sum_labels,
                                                       'ALPHADCG')
    self.assertAllClose(as_list_weights, [5.5, 5.5])
    expected_alphadcg = (
        expected_alphadcg_1 * as_list_weights[0] +
        expected_alphadcg_2 * as_list_weights[1]) / sum(as_list_weights)
    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores, weights)
    self.assertAlmostEqual(metric_.result().numpy(), expected_alphadcg,
                           places=5)
    # Test zero AlphaDCG cases.
    metric_ = metrics_lib.AlphaDCGMetric()
    metric_.update_state(labels, scores, [[0.], [0.]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

  def test_ordered_pair_accuracy(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[-1., 0., 1.], [0., 1., 2.]]
    weights = [[1.], [2.]]
    item_weights = [[1., 1., 1.], [2., 2., 3.]]

    metric_ = metrics_lib.OPAMetric()
    metric_.update_state([labels[0]], [scores[0]])
    self.assertAlmostEqual(metric_.result().numpy(), 0., places=5)

    metric_ = metrics_lib.OPAMetric()
    metric_.update_state([labels[1]], [scores[1]])
    self.assertAlmostEqual(metric_.result().numpy(), 1., places=5)

    metric_ = metrics_lib.OPAMetric()
    metric_.update_state(labels, scores)
    expected_result = (0. + 3.) / (1. + 3.)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.OPAMetric()
    metric_.update_state(labels, scores, weights)
    expected_result = (0. + 3. * 2.) / (1. + 3. * 2.)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

    metric_ = metrics_lib.OPAMetric()
    metric_.update_state(labels, scores, item_weights)
    expected_result = (0. + 2. + 3. + 3.) / (1. + 2. + 3. + 3.)
    self.assertAlmostEqual(metric_.result().numpy(), expected_result, places=5)

  def test_default_keras_metrics(self):
    default_metrics = metrics_lib.default_keras_metrics()
    self.assertLen(default_metrics, 11)
    for metric in default_metrics:
      self.assertIsInstance(metric, tf.keras.metrics.Metric)

if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
