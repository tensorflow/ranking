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

"""Tests for ranking metrics implementation."""

import math
import tensorflow as tf

from tensorflow_ranking.python import metrics_impl


def log2p1(x):
  return math.log2(1. + x)


class MetricsImplTest(tf.test.TestCase):

  def test_reset_invalid_labels(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., -1., 1.]]
      mask = [[True, False, True]]
      labels, predictions, _, _ = metrics_impl._prepare_and_validate_params(
          labels, scores, mask)
      self.assertAllClose(labels, [[0., 0., 1.]])
      self.assertAllClose(predictions, [[1., 1. - 1e-6, 2]])


class MRRMetricTest(tf.test.TestCase):

  def test_mrr_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 1.]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / 2.]])

  def test_mrr_should_be_0_when_no_rel_item(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_mrr_should_be_0_when_no_rel_item_in_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 1.]]

      metric = metrics_impl.MRRMetric(name=None, topn=1)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_mrr_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]
      labels = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

      metric_top1 = metrics_impl.MRRMetric(name=None, topn=1)
      metric_top2 = metrics_impl.MRRMetric(name=None, topn=2)
      metric_top6 = metrics_impl.MRRMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[1.], [0.], [0.]])
      self.assertAllClose(output_top2, [[1.], [1. / 2.], [0.]])
      self.assertAllClose(output_top6, [[1.], [1. / 2.], [1. / 3.]])

  def test_mrr_should_ignore_padded_labels(self):
    with tf.Graph().as_default():
      scores = [[1., 2., 3.]]
      labels = [[0., 1., -1.]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1.]])

  def test_mrr_should_ignore_masked_items(self):
    with tf.Graph().as_default():
      scores = [[1., 2., 3.]]
      labels = [[0., 1., 0.]]
      mask = [[True, True, False]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None, mask=mask)

      self.assertAllClose(output, [[1.]])

  def test_mrr_should_give_a_value_for_each_list_in_batch_inputs(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 1.]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / 2.], [1.]])

  def test_mrr_weights_should_be_average_weight_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[1., 0., 0.], [0., 1., 1.]]
      weights = [[2., 5., 1.], [1., 2., 3.]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[2.], [(2. + 3.) / 2.]])

  def test_mrr_weights_should_be_0_without_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[2., 5., 1.]]

      metric = metrics_impl.MRRMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])

  def test_mrr_weights_should_be_regardless_of_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [1., 3., 2.]]
      labels = [[1., 0., 1.], [0., 1., 1.]]
      weights = [[2., 0., 5.], [1., 4., 2.]]

      metric = metrics_impl.MRRMetric(name=None, topn=2)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(5. + 2.) / 2.], [(2. + 4.) / 2.]])


class ARPMetricTest(tf.test.TestCase):

  def test_arp_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 1.]]

      metric = metrics_impl.ARPMetric(name=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[2.]])

  def test_arp_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]

      metric = metrics_impl.ARPMetric(name=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[2.],
                                   [((1. * 2.) + (2. * 1.)) / (2. + 1.)]])

  def test_arp_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.ARPMetric(name=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_arp_should_ignore_padded_items(self):
    with tf.Graph().as_default():
      scores = [[1., 5., 4., 3., 2.]]
      labels = [[1., -1., 1., -1., 0.]]

      metric = metrics_impl.ARPMetric(name=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[2.]])

  def test_arp_should_ignore_masked_items(self):
    with tf.Graph().as_default():
      scores = [[1., 5., 4., 3., 2.]]
      labels = [[1., 0., 1., 1., 0.]]
      mask = [[True, False, True, False, True]]

      metric = metrics_impl.ARPMetric(name=None)
      output, _ = metric.compute(labels, scores, None, mask=mask)

      self.assertAllClose(output, [[2.]])

  def test_arp_should_weight_items_with_weights_and_labels(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]

      metric = metrics_impl.ARPMetric(name=None)
      output, _ = metric.compute(labels, scores, weights)

      self.assertAllClose(
          output,
          [[(2.)], [(2. * (5. / (5. + 6. * 2.)) +
                     1. * (6. * 2. / (5. + 6. * 2.)))]])

  def test_arp_weights_should_be_sum_of_weighted_labels(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]

      metric = metrics_impl.ARPMetric(name=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[3.],
                                           [5. + 6. * 2.]])


class RecallMetricTest(tf.test.TestCase):

  def test_recall_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 1.]]

      metric = metrics_impl.RecallMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1.]])

  def test_recall_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.RecallMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_recall_should_ignore_masked_items(self):
    with tf.Graph().as_default():
      scores = [[5., 4., 3., 2., 1.]]
      labels = [[0., 1., 1., 0., 1.]]
      mask = [[True, False, True, True, True]]

      metric = metrics_impl.RecallMetric(name=None, topn=3)
      output, _ = metric.compute(labels, scores, None, mask=mask)

      self.assertAllClose(output, [[1. / 2.]])

  def test_recall_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 1.]]

      metric_top1 = metrics_impl.RecallMetric(name=None, topn=1)
      metric_top2 = metrics_impl.RecallMetric(name=None, topn=2)
      metric_top6 = metrics_impl.RecallMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[0.]])
      self.assertAllClose(output_top2, [[1.]])
      self.assertAllClose(output_top6, [[1.]])

  def test_recall_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 3., 4.]]
      labels = [[1., 0., 1.], [0., 1., 1.]]

      metric = metrics_impl.RecallMetric(name=None, topn=2)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / 2.], [1.]])

  def test_recall_weights_should_be_avg_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 1., 0.]]
      weights = [[3., 9., 2.]]

      metric = metrics_impl.RecallMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 9.) / 2.]])

  def test_recall_weights_should_ignore_graded_relevance(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[4., 0., 2.]]
      weights = [[3., 9., 2.]]

      metric = metrics_impl.RecallMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 2.) / 2.]])

  def test_recall_weights_should_ignore_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 1., 0.]]
      weights = [[3., 9., 2.]]

      metric = metrics_impl.RecallMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 9.) / 2.]])

  def test_recall_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.RecallMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, None)

      self.assertAllClose(output_weights, [[0.]])


class PrecisionMetricTest(tf.test.TestCase):

  def test_precision_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 1.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / 3.]])

  def test_precision_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_precision_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.], [4., 1., 3., 2.]]
      labels = [[0., 0., 1., 1.], [0., 0., 1., 0.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[2. / 4.], [1. / 4.]])

  def test_precision_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]
      labels = [[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]

      metric_top1 = metrics_impl.PrecisionMetric(name=None, topn=1)
      metric_top2 = metrics_impl.PrecisionMetric(name=None, topn=2)
      metric_top6 = metrics_impl.PrecisionMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[1. / 1.], [0. / 1.], [0. / 1.]])
      self.assertAllClose(output_top2, [[1. / 2.], [1. / 2.], [0. / 2.]])
      self.assertAllClose(output_top6, [[2. / 3.], [1. / 3.], [1. / 3.]])

  def test_precision_should_ignore_padded_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.], [4., 1., 3., 2.]]
      labels = [[0., 0., 1., -1.], [0., -1., 1., -1.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / 3.], [1. / 2.]])

  def test_precision_should_ignore_masked_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.], [4., 1., 3., 2.]]
      labels = [[0., 0., 1., 0.], [0., 1., 1., 0.]]
      mask = [[True, True, True, False], [True, False, True, False]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None, mask=mask)

      self.assertAllClose(output, [[1. / 3.], [1. / 2.]])

  def test_precision_weights_should_be_avg_of_weights_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[13., 7., 29.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(13. + 29.) / 2.]])

  def test_precision_weights_should_ignore_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 1., 0.]]
      weights = [[3., 7., 15.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 7.) / 2.]])

  def test_precision_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[3., 7., 15.]]

      metric = metrics_impl.PrecisionMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])


class MeanAveragePrecisionMetricTest(tf.test.TestCase):

  def test_map_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 1., 0.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. / 2.) / 1.]])

  def test_map_should_treat_graded_relevance_as_binary_relevance(self):
    with tf.Graph().as_default():
      scores = [[3., 4., 1., 2.]]
      labels = [[0., 2., 1., 3.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. + 2. / 3. + 3. / 4.) / 3.]])

  def test_map_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_map_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 3., 2.]]
      labels = [[0., 0., 1.], [0., 1., 1.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. / 2.) / 1.],
                                   [(1. / 1. + 2. / 2.) / 2.]])

  def test_map_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]
      labels = [[1., 0., 2.], [0., 1., 0.], [0., 0., 1.]]

      metric_top1 = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=1)
      metric_top2 = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=2)
      metric_top6 = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[1. / 2.],
                                        [0. / 1.],
                                        [0. / 1.]])
      self.assertAllClose(output_top2, [[1. / 2.],
                                        [(1. / 2.) / 1.],
                                        [0. / 1.]])
      self.assertAllClose(output_top6, [[(1. + 2. / 3.) / 2.],
                                        [(1. / 2.) / 1.],
                                        [(1. / 3.) / 1.]])

  def test_map_weights_should_be_avg_of_weights_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[13., 7., 29.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(13. + 29.) / 2.]])

  def test_map_weights_should_ignore_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 1., 0.]]
      weights = [[3., 7., 15.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 7.) / 2.]])

  def test_map_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[3., 7., 15.]]

      metric = metrics_impl.MeanAveragePrecisionMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])


class NDCGMetricTest(tf.test.TestCase):

  def test_ndcg_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 1., 0.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      dcg = 1. / log2p1(2.)
      max_dcg = 1. / log2p1(1.)
      self.assertAllClose(output, [[dcg / max_dcg]])

  def test_ndcg_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_ndcg_should_operate_on_graded_relevance(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 3., 1., 0.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      dcg = (2. ** 3. - 1.) / log2p1(2.) + 1. / log2p1(3.)
      max_dcg = (2. ** 3. - 1.) / log2p1(1.) + 1. / log2p1(2.)
      self.assertAllClose(output, [[dcg / max_dcg]])

  def test_ndcg_should_operate_on_graded_relevance_with_custom_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 3., 1., 0.]]
      gain_fn = lambda label: label / 2.

      metric = metrics_impl.NDCGMetric(name=None, topn=None, gain_fn=gain_fn)
      output, _ = metric.compute(labels, scores, None)

      dcg = (3. / 2.) / log2p1(2.) + (1. / 2.) / log2p1(3.)
      max_dcg = (3. / 2.) / log2p1(1.) + (1. / 2.) / log2p1(2.)
      self.assertAllClose(output, [[dcg / max_dcg]])

  def test_ndcg_should_use_custom_rank_discount_fn(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 3., 1., 0.]]
      rank_discount_fn = lambda rank: 1.0 / (rank + 10.0)

      metric = metrics_impl.NDCGMetric(name=None, topn=None,
                                       rank_discount_fn=rank_discount_fn)
      output, _ = metric.compute(labels, scores, None)

      dcg = (2. ** 3. - 1.) / (2. + 10.) + 1. / (3. + 10.)
      max_dcg = (2. ** 3. - 1.) / (1. + 10.) + 1. / (2. + 10.)
      self.assertAllClose(output, [[dcg / max_dcg]])

  def test_ndcg_should_ignore_padded_items(self):
    with tf.Graph().as_default():
      scores = [[1., 4., 3., 2.]]
      labels = [[2., -1., 1., 0.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      dcg = (2. ** 2. - 1.) / log2p1(3.) + 1. / log2p1(1.)
      max_dcg = (2. ** 2. - 1.) / log2p1(1.) + 1. / log2p1(2.)
      self.assertAllClose(output, [[dcg / max_dcg]])

  def test_ndcg_should_ignore_masked_items(self):
    with tf.Graph().as_default():
      scores = [[1., 4., 3., 2.]]
      labels = [[2., 2., 1., 0.]]
      mask = [[True, False, True, True]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None, mask=mask)

      dcg = (2. ** 2. - 1.) / log2p1(3.) + 1. / log2p1(1.)
      max_dcg = (2. ** 2. - 1.) / log2p1(1.) + 1. / log2p1(2.)
      self.assertAllClose(output, [[dcg / max_dcg]])

  def test_ndcg_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 1., 2.]]
      labels = [[0., 1., 0.], [1., 1., 0.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      dcg = [1. / log2p1(2.), 1. / log2p1(1.) + 1. / log2p1(3.)]
      max_dcg = [1. / log2p1(1.), 1. / log2p1(1.) + 1. / log2p1(2.)]
      self.assertAllClose(output,
                          [[dcg[0] / max_dcg[0]], [dcg[1] / max_dcg[1]]])

  def test_ndcg_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]
      labels = [[1., 0., 2.], [0., 1., 0.], [0., 0., 1.]]

      metric_top1 = metrics_impl.NDCGMetric(name=None, topn=1)
      metric_top2 = metrics_impl.NDCGMetric(name=None, topn=2)
      metric_top6 = metrics_impl.NDCGMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      max_dcg_top1 = [(2. ** 2. - 1.) / log2p1(1.),
                      1. / log2p1(1.),
                      1. / log2p1(1.)]
      max_dcg = [(2. ** 2. - 1.) / log2p1(1.) + 1. / log2p1(2.),
                 1. / log2p1(1.),
                 1. / log2p1(1.)]

      self.assertAllClose(output_top1,
                          [[(1. / log2p1(1.)) / max_dcg_top1[0]],
                           [0. / max_dcg_top1[1]],
                           [0. / max_dcg_top1[2]]])
      self.assertAllClose(output_top2,
                          [[(1. / log2p1(1.)) / max_dcg[0]],
                           [(1. / log2p1(2.)) / max_dcg[1]],
                           [0. / max_dcg[2]]])
      self.assertAllClose(output_top6,
                          [[(1. / log2p1(1.) + (2. ** 2. - 1.) / log2p1(3.)) /
                            max_dcg[0]],
                           [(1. / log2p1(2.)) / max_dcg[1]],
                           [(1. / log2p1(3.)) / max_dcg[2]]])

  def test_ndcg_weights_should_be_average_of_weighted_gain(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[3., 7., 9.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(
          output_weights,
          [[(1. * 3. + (2. ** 2. - 1.) * 9.) / (1. + (2. ** 2. - 1.))]])

  def test_ndcg_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[2., 4., 4.]]

      metric = metrics_impl.NDCGMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])

  def test_ndcg_weights_should_use_custom_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[3., 7., 9.]]
      gain_fn = lambda label: label + 5.

      metric = metrics_impl.NDCGMetric(name=None, topn=None, gain_fn=gain_fn)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(
          output_weights,
          [[((1. + 5.) * 3. + (0. + 5.) * 7. + (2. + 5.) * 9.) /
            ((1. + 5.) + (0. + 5.) + (2. + 5.))]])


class DCGMetricTest(tf.test.TestCase):

  def test_dcg_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 1., 0.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / log2p1(2.)]])

  def test_dcg_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_dcg_should_operate_on_graded_relevance(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 3., 1., 0.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[(2. ** 3. - 1.) / log2p1(2.) + 1. / log2p1(3.)]])

  def test_dcg_should_operate_on_graded_relevance_with_custom_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 3., 1., 0.]]
      gain_fn = lambda label: label / 2.

      metric = metrics_impl.DCGMetric(name=None, topn=None, gain_fn=gain_fn)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[(3. / 2.) / log2p1(2.) + (1. / 2.) / log2p1(3.)]])

  def test_dcg_should_use_custom_rank_discount_fn(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 3., 1., 0.]]
      rank_discount_fn = lambda rank: 1.0 / (rank + 10.0)

      metric = metrics_impl.DCGMetric(name=None, topn=None,
                                      rank_discount_fn=rank_discount_fn)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[(2. ** 3. - 1.) / (2. + 10.) + 1. / (3. + 10.)]])

  def test_dcg_should_ignore_padded_items(self):
    with tf.Graph().as_default():
      scores = [[1., 4., 3., 2.]]
      labels = [[2., -1., 1., 0.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[(2. ** 2. - 1.) / log2p1(3.) + 1. / log2p1(1.)]])

  def test_dcg_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 1., 2.]]
      labels = [[0., 1., 0.], [1., 1., 0.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / log2p1(2.)],
                                   [1. / log2p1(1.) + 1. / log2p1(3.)]])

  def test_dcg_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]
      labels = [[1., 0., 2.], [0., 1., 0.], [0., 0., 1.]]

      metric_top1 = metrics_impl.DCGMetric(name=None, topn=1)
      metric_top2 = metrics_impl.DCGMetric(name=None, topn=2)
      metric_top6 = metrics_impl.DCGMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[(1. / log2p1(1.))], [0.], [0.]])
      self.assertAllClose(output_top2,
                          [[(1. / log2p1(1.))], [(1. / log2p1(2.))], [0.]])
      self.assertAllClose(output_top6,
                          [[(1. / log2p1(1.) + (2. ** 2. - 1.) / log2p1(3.))],
                           [(1. / log2p1(2.))],
                           [(1. / log2p1(3.))]])

  def test_dcg_weights_should_be_average_of_weighted_gain(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[3., 7., 9.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(
          output_weights,
          [[(1. * 3. + (2. ** 2. - 1.) * 9.) / (1. + (2. ** 2. - 1.))]])

  def test_dcg_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[2., 4., 4.]]

      metric = metrics_impl.DCGMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])

  def test_dcg_weights_should_use_custom_gain_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[4., 1., 9.]]
      gain_fn = lambda label: label + 3.

      metric = metrics_impl.DCGMetric(name=None, topn=None, gain_fn=gain_fn)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(
          output_weights,
          [[((1. + 3.) * 4. + (0. + 3.) * 1. + (2. + 3.) * 9.) /
            ((1. + 3.) + (0. + 3.) + (2. + 3.))]])


class OPAMetricTest(tf.test.TestCase):

  def test_opa_should_return_correct_pair_matrix(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 1., 0.]]

      metric = metrics_impl.OPAMetric(name=None)
      output, output_weights = metric.compute(labels, scores, None)

      # The correctly ordered pair is:
      # scores[1] > scores[2]
      self.assertAllClose(output, [[1. / 2.]])
      self.assertAllClose(output_weights, [[2.]])

  def test_opa_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.OPAMetric(name=None)
      output, output_weights = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])
      self.assertAllClose(output_weights, [[0.]])

  def test_opa_should_operate_on_graded_relevance(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[1., 3., 0., 1.]]

      metric = metrics_impl.OPAMetric(name=None)
      output, output_weights = metric.compute(labels, scores, None)

      # The correctly ordered pairs are:
      # scores[0] > scores[2]
      # scores[1] > scores[2], scores[1] > scores[3]
      self.assertAllClose(output, [[3. / 5.]])
      self.assertAllClose(output_weights, [[5.]])

  def test_opa_should_ignore_padded_items(self):
    with tf.Graph().as_default():
      scores = [[4., 1., 2., 3.]]
      labels = [[2., -1., 1., 0.]]

      metric = metrics_impl.OPAMetric(name=None)
      output, output_weights = metric.compute(labels, scores, None)

      # The correctly ordered pairs are:
      # scores[0] > scores[2], scores[0] > scores[3]
      self.assertAllClose(output, [[2. / 3.]])
      self.assertAllClose(output_weights, [[3.]])

  def test_opa_should_return_correct_pair_matrix_per_list(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 1., 2.]]
      labels = [[0., 1., 0.], [1., 0., 1.]]

      metric = metrics_impl.OPAMetric(name=None)
      output, output_weights = metric.compute(labels, scores, None)

      # The correctly ordered pairs are:
      # list 1: scores[1] > scores[2]
      # list 2: scores[0] > scores[1], scores[2] > scores[1]
      self.assertAllClose(output, [[1. / 2.], [2. / 2.]])
      self.assertAllClose(output_weights, [[2.], [2.]])

  def test_opa_should_weight_pairs_with_weights(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[3., 7., 9.]]

      metric = metrics_impl.OPAMetric(name=None)
      output, _ = metric.compute(labels, scores, weights)

      # The correctly ordered pair is:
      # scores[2] > scores[0] (with weight 9.)
      self.assertAllClose(output, [[9. / (9. + 9. + 3.)]])

  def test_opa_weights_should_be_sum_of_pair(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[3., 7., 9.]]

      metric = metrics_impl.OPAMetric(name=None)
      _, output_weights = metric.compute(labels, scores, weights)

      # The OPA weights are based on the label pairs:
      # labels[0] > labels[1] (with weight 3.)
      # labels[2] > labels[0] (with weight 9.)
      # labels[2] > labels[1] (with weight 9.)
      self.assertAllClose(output_weights, [[9. + 9. + 3.]])

  def test_opa_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[2., 4., 4.]]

      metric = metrics_impl.OPAMetric(name=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])


class PrecisionIAMetricTest(tf.test.TestCase):

  def test_precisionia_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0.], [1., 0.], [0., 1.]]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[2. / (2. * 3.)]])

  def test_precisionia_should_handle_single_subtopic(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0.], [1.], [0.]]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1. / (1. * 3.)]])

  def test_precisionia_should_handle_multiple_subtopics(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0., 1., 0.], [1., 1., 1., 1.], [0., 1., 1., 0.]]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[7. / (4. * 3.)]])

  def test_precisionia_should_ignore_subtopics_without_rel(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.]]
      labels = [[[0., 0.], [0., 1.], [0., 1.], [0., 0.]]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[2. / (1. * 4.)]])

  def test_precisionia_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_precisionia_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.], [4., 1., 3., 2.]]
      labels = [[[0., 0.], [0., 0.], [1., 1.], [1., 0.]],
                [[1., 0.], [1., 1.], [1., 0.], [0., 1.]]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[3. / (2. * 4.)], [5. / (2. * 4.)]])

  def test_precisionia_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.], [3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]
      labels = [[[1., 1.], [0., 0.], [1., 0.]],
                [[0., 0.], [0., 1.], [1., 0.]],
                [[0., 1.], [0., 0.], [1., 0.]],
                [[1., 1.], [1., 1.], [1., 1.]]]

      metric_top1 = metrics_impl.PrecisionIAMetric(name=None, topn=1)
      metric_top2 = metrics_impl.PrecisionIAMetric(name=None, topn=2)
      metric_top6 = metrics_impl.PrecisionIAMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[2. / (2. * 1.)],
                                        [0. / (2. * 1.)],
                                        [1. / (2. * 1.)],
                                        [2. / (2. * 1.)]])
      self.assertAllClose(output_top2, [[2. / (2. * 2.)],
                                        [1. / (2. * 2.)],
                                        [1. / (2. * 2.)],
                                        [4. / (2. * 2.)]])
      self.assertAllClose(output_top6, [[3. / (2. * 3.)],
                                        [2. / (2. * 3.)],
                                        [2. / (2. * 3.)],
                                        [6. / (2. * 3.)]])

  def test_precisionia_weights_should_be_avg_of_weights_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 1.], [0., 0.], [1., 1.]]]
      weights = [[3., 7., 9.]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 9.) / 2.]])

  def test_precisionia_weights_should_ignore_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[1., 1.], [1., 0.], [0., 0.]]]
      weights = [[3., 4., 5.]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 4.) / 2.]])

  def test_precisionia_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]]]
      weights = [[3., 7., 2.]]

      metric = metrics_impl.PrecisionIAMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])


class AlphaDCGMetricTest(tf.test.TestCase):

  def test_alphadcg_should_be_single_value(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0.], [1., 0.], [0., 1.]]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                    (1. * (1. - 0.5) ** 0.) / log2p1(2.)]])

  def test_alphadcg_should_handle_single_subtopic(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0.], [0.], [1.]]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. * (1. - 0.5) ** 0.) / log2p1(2.)]])

  def test_alphadcg_should_handle_multiple_subtopics(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 1., 0., 0.], [1., 1., 0., 1.], [0., 1., 1., 0.]]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                    (1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                    (1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                    (1. * (1. - 0.5) ** 1.) / log2p1(2.) +
                                    (1. * (1. - 0.5) ** 0.) / log2p1(2.) +
                                    (1. * (1. - 0.5) ** 2.) / log2p1(3.)]])

  def test_alphadcg_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_alphadcg_should_be_single_value_per_list(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.], [4., 1., 3., 2.]]
      labels = [[[0., 0.], [0., 0.], [1., 1.], [1., 0.]],
                [[1., 0.], [1., 1.], [1., 0.], [0., 1.]]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1.) / log2p1(1.) +
                                    (1. * (1. - 0.5) ** 0.) / log2p1(3.) +
                                    (1. * (1. - 0.5) ** 1.) / log2p1(3.)],
                                   [(1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                    (1. * (1. - 0.5) ** 1.) / log2p1(2.) +
                                    (1. * (1. - 0.5) ** 0.) / log2p1(3.) +
                                    (1. * (1. - 0.5) ** 1.) / log2p1(4.) +
                                    (1. * (1. - 0.5) ** 2.) / log2p1(4.)]])

  def test_alphadcg_should_handle_custom_alpha(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.]]
      labels = [[[1., 1.], [0., 1.], [0., 1.], [1., 0.]]]

      metric_alpha2 = metrics_impl.AlphaDCGMetric(name=None, topn=None,
                                                  alpha=0.2)
      metric_alpha95 = metrics_impl.AlphaDCGMetric(name=None, topn=None,
                                                   alpha=0.95)
      output_alpha2, _ = metric_alpha2.compute(labels, scores, None)
      output_alpha95, _ = metric_alpha95.compute(labels, scores, None)

      self.assertAllClose(output_alpha2,
                          [[(1. * (1. - 0.2) ** 0.) / log2p1(1.) +
                            (1. * (1. - 0.2) ** 0.) / log2p1(2.) +
                            (1. * (1. - 0.2) ** 1.) / log2p1(3.) +
                            (1. * (1. - 0.2) ** 1.) / log2p1(4.) +
                            (1. * (1. - 0.2) ** 2.) / log2p1(4.)]])
      self.assertAllClose(output_alpha95,
                          [[(1. * (1. - 0.95) ** 0.) / log2p1(1.) +
                            (1. * (1. - 0.95) ** 0.) / log2p1(2.) +
                            (1. * (1. - 0.95) ** 1.) / log2p1(3.) +
                            (1. * (1. - 0.95) ** 1.) / log2p1(4.) +
                            (1. * (1. - 0.95) ** 2.) / log2p1(4.)]])

  def test_alphadcg_should_handle_custom_rank_discount_fn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2., 4.]]
      labels = [[[1., 0.], [1., 1.], [0., 1.], [1., 0.]]]
      rank_discount_fn = lambda rank: 1. / (10. + rank)

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None,
                                           rank_discount_fn=rank_discount_fn)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(1. * (1. - 0.5) ** 0.) / (10. + 1.) +
                                    (1. * (1. - 0.5) ** 0.) / (10. + 2.) +
                                    (1. * (1. - 0.5) ** 1.) / (10. + 2.) +
                                    (1. * (1. - 0.5) ** 1.) / (10. + 3.) +
                                    (1. * (1. - 0.5) ** 2.) / (10. + 4.)]])

  def test_alphadcg_should_handle_topn(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1., 4., 5.]]
      labels = [[[1., 0.], [0., 0.], [1., 0.], [1., 1.], [0., 1.]]]

      metric_top1 = metrics_impl.AlphaDCGMetric(name=None, topn=1)
      metric_top2 = metrics_impl.AlphaDCGMetric(name=None, topn=2)
      metric_top6 = metrics_impl.AlphaDCGMetric(name=None, topn=6)
      output_top1, _ = metric_top1.compute(labels, scores, None)
      output_top2, _ = metric_top2.compute(labels, scores, None)
      output_top6, _ = metric_top6.compute(labels, scores, None)

      self.assertAllClose(output_top1, [[(1. * (1. - 0.5) ** 0.) / log2p1(1.)]])
      self.assertAllClose(output_top2, [[(1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                         (1. * (1. - 0.5) ** 0.) / log2p1(2.) +
                                         (1. * (1. - 0.5) ** 1.) / log2p1(2.)]])
      self.assertAllClose(output_top6, [[(1. * (1. - 0.5) ** 0.) / log2p1(1.) +
                                         (1. * (1. - 0.5) ** 0.) / log2p1(2.) +
                                         (1. * (1. - 0.5) ** 1.) / log2p1(2.) +
                                         (1. * (1. - 0.5) ** 1.) / log2p1(3.) +
                                         (1. * (1. - 0.5) ** 2.) / log2p1(5.)]])

  def test_alphadcg_weights_should_be_avg_of_weights_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 1.], [0., 0.], [1., 1.]]]
      weights = [[3., 7., 9.]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 9.) / 2.]])

  def test_alphadcg_weights_should_ignore_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[1., 1.], [1., 0.], [0., 0.]]]
      weights = [[3., 4., 5.]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 4.) / 2.]])

  def test_alphadcg_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[[0., 0.], [0., 0.], [0., 0.]]]
      weights = [[3., 7., 2.]]

      metric = metrics_impl.AlphaDCGMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])


class BPrefMetricTest(tf.test.TestCase):

  def test_bpref_should_be_a_single_value(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 1., 0., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[1. / 2. * ((1. - 1. / 2.) + (1. - 2. / 2.))]])

  def test_bpref_should_convert_graded_relevance_to_binary(self):
    with tf.Graph().as_default():
      scores = [[4., 3., 2., 1.]]
      labels = [[0., 1., 0., 2.]]  # should be equivalent to [[0., 1., 0., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[1. / 2. * ((1. - 1. / 2.) + (1. - 2. / 2.))]])

  def test_bpref_should_be_zero_when_only_irrelevant_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 0., 0.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_trec_bpref_should_be_one_when_only_relevant_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[1., 1., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1.]])

  def test_non_trec_bpref_should_be_one_when_only_relevant_items(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[1., 1., 1.]]

      metric = metrics_impl.BPrefMetric(
          name=None, topn=None, use_trec_version=False)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[1.]])

  def test_bpref_should_be_zero_without_input_items(self):
    with tf.Graph().as_default():
      scores = [[]]
      labels = [[]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_trec_bpref_divides_with_min_and_is_0_when_one_irrelevant_first(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 1., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_bpref_divides_with_r_when_use_trec_version_is_false(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1.]]
      labels = [[0., 1., 1.]]

      metric = metrics_impl.BPrefMetric(
          name=None, topn=None, use_trec_version=False)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.5]])

  def test_bpref_should_be_0_when_no_rel_item_in_topn_but_relevant_later(self):
    with tf.Graph().as_default():
      scores = [[3., 2., 1., 0.]]
      labels = [[0., 0., 0., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=3)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[0.]])

  def test_non_trec_bpref_should_handle_topn(self):
    with tf.Graph().as_default():
      # This is the example case in bpref_bug in TREC_EVAL-8.0+
      scores = [[5., 4., 3., 2., 1., 0., 0., 0., 0., 0.]]
      labels = [[0., 1., 1., 1., 1., 0., 0., 0., 1., 1.]]

      metric = metrics_impl.BPrefMetric(
          name=None, topn=5, use_trec_version=False)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(4. * (1. - 1. / 6.)) / 6.]])

  def test_trec_bpref_should_handle_topn(self):
    with tf.Graph().as_default():
      # This is the example case in bpref_bug in TREC_EVAL-8.0+
      scores = [[5., 4., 3., 2., 1., 0., 0., 0., 0., 0.]]
      labels = [[0., 1., 1., 1., 1., 0., 0., 0., 1., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=5)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output, [[(4. * (1. - 1. / 4.)) / 6.]])  # = 0.5

  def test_bpref_should_ignore_padded_items(self):
    with tf.Graph().as_default():
      scores = [[6., 5., 4., 3., 2., 1.]]
      labels = [[-1., 0., -1., 1., 0., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose(output,
                          [[1. / 2. * ((1. - 1. / 2.) + (1. - 2. / 2.))]])

  def test_bpref_weights_should_be_avg_of_weights_of_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 0., 2.]]
      weights = [[13., 7., 29.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(13. + 29.) / 2.]])

  def test_bpref_weights_should_ignore_topn(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[1., 1., 0.]]
      weights = [[3., 7., 15.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=1)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[(3. + 7.) / 2.]])

  def test_bpref_weights_should_be_0_when_no_rel_items(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      labels = [[0., 0., 0.]]
      weights = [[3., 7., 15.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      _, output_weights = metric.compute(labels, scores, weights)

      self.assertAllClose(output_weights, [[0.]])

  def test_bpref_should_give_a_value_for_each_list_in_batch_inputs(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 1.]]

      metric = metrics_impl.BPrefMetric(name=None, topn=None)
      output, _ = metric.compute(labels, scores, None)

      self.assertAllClose([[0.], [1.]], output)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
