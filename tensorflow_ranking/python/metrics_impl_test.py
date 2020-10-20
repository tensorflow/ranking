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

import tensorflow as tf

from tensorflow_ranking.python import metrics_impl


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


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
