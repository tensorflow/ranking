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

"""Tests for ranking head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_ranking.python import head as ranking_head
from tensorflow_ranking.python import metrics as metrics_lib


def _initialize_variables(test_case, scaffold):
  """Initializes variables for a session.

  Args:
    test_case: A TensorFlowTestCase object.
    scaffold: A train.Scaffold object.
  """
  scaffold.finalize()
  test_case.assertIsNone(scaffold.init_feed_dict)
  test_case.assertIsNone(scaffold.init_fn)
  scaffold.init_op.run()
  scaffold.ready_for_local_init_op.eval()
  scaffold.local_init_op.run()
  scaffold.ready_op.eval()
  test_case.assertIsNotNone(scaffold.saver)


def _make_loss_fn(weights_feature_name=None):
  """Make a fake loss function."""

  def _loss_fn(labels, logits, features):
    """A fake loss function."""
    logits = tf.convert_to_tensor(value=logits)
    labels = tf.cast(labels, dtype=tf.float32)
    weights = features[
        weights_feature_name] if weights_feature_name is not None else 1.
    loss = tf.reduce_sum(input_tensor=logits -
                         labels) * tf.reduce_sum(input_tensor=weights)
    return loss

  return _loss_fn


class UtilTest(tf.test.TestCase):

  def test_get_train_op(self):
    with tf.Graph().as_default():
      logits = tf.cast([[1.]], tf.float32)
      labels = tf.cast([[0.]], tf.float32)
      loss = tf.abs(tf.reduce_sum(input_tensor=logits - labels))

      def _train_op_fn(loss):
        with tf.control_dependencies((tf.compat.v1.assert_near(
            tf.cast(loss, dtype=tf.float32), 1.0, name='assert_loss'),)):
          return tf.constant(b'train_op_fn')

      class _Optimizer(object):

        def minimize(self, loss, global_step):
          del global_step
          with tf.control_dependencies((tf.compat.v1.assert_equal(
              tf.cast(loss, dtype=tf.float32), 1.0, name='assert_loss'),)):
            return tf.constant(b'optimizer')

      train_op = ranking_head._get_train_op(loss, _train_op_fn, None)
      self.assertIsNotNone(train_op)
      with self.cached_session() as sess:
        train_result = sess.run(train_op)
        self.assertEqual(b'train_op_fn', train_result)

      train_op = ranking_head._get_train_op(loss, None, _Optimizer())
      self.assertIsNotNone(train_op)
      with self.cached_session() as sess:
        train_result = sess.run(train_op)
        self.assertEqual(b'optimizer', train_result)

      with self.assertRaisesRegexp(
          ValueError, r'train_op_fn and optimizer cannot both be set.'):
        ranking_head._get_train_op(loss, _train_op_fn, _Optimizer())
      with self.assertRaisesRegexp(
          ValueError, r'train_op_fn and optimizer cannot both be None.'):
        ranking_head._get_train_op(loss, None, None)


class RankingHeadTest(tf.test.TestCase):

  def setUp(self):
    super(RankingHeadTest, self).setUp()
    self._default_logits = [[1., 3., 2.], [1., 2., 3.]]
    self._default_labels = [[0., 0., 1.], [0., 0., 2.]]
    self._default_loss = 9.
    self._default_weights = [1.] * 3
    self._default_weights_feature_name = 'weights'
    self._default_weighted_loss = 27

  def test_name(self):
    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(), name='fake_head')
    self.assertEqual('fake_head', head.name)

  def test_labels_and_logits_metrics(self):
    head = ranking_head.create_ranking_head(loss_fn=_make_loss_fn())
    with tf.Graph().as_default():
      logits = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 0., 2.]]
      metrics_dict = head._labels_and_logits_metrics(labels, logits)
      self.assertCountEqual(['labels_mean', 'logits_mean'], metrics_dict)
      with self.cached_session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        for (metric_op,
             update_op), value in [(metrics_dict['labels_mean'], 0.5),
                                   (metrics_dict['logits_mean'], 2.0)]:
          sess.run(update_op)
          self.assertAlmostEqual(sess.run(metric_op), value, places=5)

  def test_predict(self):
    with tf.Graph().as_default():
      head = ranking_head.create_ranking_head(loss_fn=_make_loss_fn())
      logits = [[1., 3.], [1., 2.]]
      spec = head.create_estimator_spec(
          features={}, mode=tf.estimator.ModeKeys.PREDICT, logits=logits)

      # Assert spec contains expected tensors.
      self.assertIsNone(spec.loss)
      self.assertEqual({}, spec.eval_metric_ops)
      self.assertIsNone(spec.train_op)
      self.assertItemsEqual(
          (ranking_head._DEFAULT_SERVING_KEY, 'regression', 'predict'),
          spec.export_outputs.keys())

      # Assert predictions.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        predictions = sess.run(spec.predictions)
        self.assertAllClose(logits, predictions)
        self.assertAllClose(
            logits,
            sess.run(
                spec.export_outputs[ranking_head._DEFAULT_SERVING_KEY].value))

  def test_eval(self):
    with tf.Graph().as_default():
      metric_fns = {
          'metric/precision@1':
              metrics_lib.make_ranking_metric_fn(
                  metrics_lib.RankingMetricKey.PRECISION, topn=1),
      }
      head = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), eval_metric_fns=metric_fns)

      # Create estimator spec.
      spec = head.create_estimator_spec(
          features={},
          mode=tf.estimator.ModeKeys.EVAL,
          logits=self._default_logits,
          labels=self._default_labels)

      expected_metrics = [
          'labels_mean',
          'logits_mean',
          'metric/precision@1',
      ]

      # Assert spec contains expected tensors.
      self.assertIsNotNone(spec.loss)
      self.assertIsNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)
      self.assertItemsEqual(expected_metrics, spec.eval_metric_ops.keys())

      # Assert predictions, loss, and metrics.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        update_ops = {
            k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops
        }
        loss, metrics = sess.run((spec.loss, update_ops))
        self.assertAllClose(self._default_loss, loss)
        self.assertItemsEqual(expected_metrics, metrics.keys())

  def test_train_create_loss(self):
    with tf.Graph().as_default():
      head = ranking_head.create_ranking_head(loss_fn=_make_loss_fn())
      # Create loss.
      training_loss = head.create_loss(
          features={},
          mode=tf.estimator.ModeKeys.TRAIN,
          logits=self._default_logits,
          labels=self._default_labels)
      with self.cached_session():
        _initialize_variables(self, tf.compat.v1.train.Scaffold())
        self.assertAllClose(self._default_loss, training_loss.eval())

  def test_train(self):
    with tf.Graph().as_default():
      expected_train_result = b'my_train_op'

      def _train_op_fn(loss):
        with tf.control_dependencies((tf.compat.v1.assert_near(
            tf.cast(self._default_loss, dtype=tf.float32),
            tf.cast(loss, dtype=tf.float32),
            name='assert_loss'),)):
          return tf.constant(expected_train_result)

      head = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), train_op_fn=_train_op_fn)
      # Create estimator spec.
      spec = head.create_estimator_spec(
          features={},
          mode=tf.estimator.ModeKeys.TRAIN,
          logits=self._default_logits,
          labels=self._default_labels)

      # Assert spec contains expected tensors.
      self.assertIsNotNone(spec.loss)
      self.assertEqual({}, spec.eval_metric_ops)
      self.assertIsNotNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)

      # Assert predictions, loss, and train_op.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        loss, train_result = sess.run((spec.loss, spec.train_op))
        self.assertAllClose(self._default_loss, loss)
        self.assertEqual(expected_train_result, train_result)

  def test_train_with_regularization_losses(self):
    with tf.Graph().as_default():
      regularization_losses = [1.5, 0.5]
      expected_regularization_loss = 2.

      expected_train_result = b'my_train_op'
      expected_loss = expected_regularization_loss + self._default_loss

      def _train_op_fn(loss):
        with tf.control_dependencies((tf.compat.v1.assert_equal(
            tf.cast(expected_loss, dtype=tf.float32),
            tf.cast(loss, dtype=tf.float32),
            name='assert_loss'),)):
          return tf.constant(expected_train_result)

      head = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), train_op_fn=_train_op_fn)

      # Create estimator spec.
      spec = head.create_estimator_spec(
          features={},
          mode=tf.estimator.ModeKeys.TRAIN,
          logits=self._default_logits,
          labels=self._default_labels,
          regularization_losses=regularization_losses)

      # Assert predictions, loss, and train_op.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        loss, train_result = sess.run((spec.loss, spec.train_op))
        self.assertAllClose(expected_loss, loss)
        self.assertEqual(expected_train_result, train_result)

  def test_multi_dim_weighted_train(self):
    with tf.Graph().as_default():
      weights_feature_name = self._default_weights_feature_name

      def _train_op_fn(loss):
        return loss

      head = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(weights_feature_name), train_op_fn=_train_op_fn)
      # Create estimator spec.
      spec = head.create_estimator_spec(
          features={weights_feature_name: self._default_weights},
          mode=tf.estimator.ModeKeys.TRAIN,
          logits=self._default_logits,
          labels=self._default_labels)

      # Assert predictions, loss, and train_op.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        loss, train_result = sess.run((spec.loss, spec.train_op))
        self.assertAllClose(self._default_weighted_loss, loss)
        self.assertAllClose(self._default_weighted_loss, train_result)

  def test_multi_dim_weighted_eval(self):
    with tf.Graph().as_default():
      weights_feature_name = self._default_weights_feature_name
      metric_fns = {
          'metric/precision@1':
              metrics_lib.make_ranking_metric_fn(
                  metrics_lib.RankingMetricKey.PRECISION, topn=1),
      }
      head = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(weights_feature_name),
          eval_metric_fns=metric_fns)

      weights = self._default_weights

      # Create estimator spec.
      spec = head.create_estimator_spec(
          features={weights_feature_name: weights},
          mode=tf.estimator.ModeKeys.EVAL,
          logits=self._default_logits,
          labels=self._default_labels)

      expected_metrics = [
          'labels_mean',
          'logits_mean',
          'metric/precision@1',
      ]

      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        update_ops = {
            k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops
        }
        loss, metrics = sess.run((spec.loss, update_ops))
        self.assertAllClose(self._default_weighted_loss, loss)
        self.assertItemsEqual(expected_metrics, metrics.keys())


class MultiRankingHeadTest(tf.test.TestCase):

  def test_predict(self):
    with tf.Graph().as_default():
      head1 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), name='head1')
      head2 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), name='head2')
      multi_head = ranking_head.create_multi_ranking_head([head1, head2])
      logits = {
          'head1': tf.convert_to_tensor(value=[[1., 3.], [1., 2.]]),
          'head2': tf.convert_to_tensor(value=[[2., 3.], [2., 2.]]),
      }
      spec = multi_head.create_estimator_spec(
          features={}, mode=tf.estimator.ModeKeys.PREDICT, logits=logits)

      # Assert spec contains expected tensors.
      self.assertIsNone(spec.loss)
      self.assertEqual({}, spec.eval_metric_ops)
      self.assertIsNone(spec.train_op)
      self.assertCountEqual([
          ranking_head._DEFAULT_SERVING_KEY, 'predict', 'head1',
          'head1/regression', 'head1/predict', 'head2', 'head2/regression',
          'head2/predict'
      ], spec.export_outputs.keys())

      # Assert predictions.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        predictions = sess.run(spec.predictions)
        self.assertAllClose(logits['head1'], predictions['head1'])
        self.assertAllClose(logits['head2'], predictions['head2'])
        self.assertAllClose(
            logits['head1'],
            sess.run(
                spec.export_outputs[ranking_head._DEFAULT_SERVING_KEY].value))

  def test_eval(self):
    with tf.Graph().as_default():
      metric_fns = {
          'metric/precision@1':
              metrics_lib.make_ranking_metric_fn(
                  metrics_lib.RankingMetricKey.PRECISION, topn=1),
      }
      head1 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), eval_metric_fns=metric_fns, name='head1')
      head2 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), eval_metric_fns=metric_fns, name='head2')
      multi_head = ranking_head.create_multi_ranking_head([head1, head2])

      logits = {
          'head1': tf.convert_to_tensor(value=[[1., 3.], [1., 2.]]),
          'head2': tf.convert_to_tensor(value=[[2., 3.], [2., 2.]]),
      }
      labels = {
          'head1': tf.convert_to_tensor(value=[[0., 1.], [0., 2.]]),
          'head2': tf.convert_to_tensor(value=[[0., 1.], [0., 2.]]),
      }
      spec = multi_head.create_estimator_spec(
          features={},
          mode=tf.estimator.ModeKeys.EVAL,
          logits=logits,
          labels=labels)

      expected_metrics = [
          'head1/labels_mean',
          'head1/logits_mean',
          'head1/metric/precision@1',
          'head2/labels_mean',
          'head2/logits_mean',
          'head2/metric/precision@1',
      ]

      # Assert spec contains expected tensors.
      self.assertIsNotNone(spec.loss)
      self.assertIsNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)
      self.assertCountEqual(spec.eval_metric_ops.keys(), expected_metrics)

      # Assert predictions, loss, and metrics.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        update_ops = {
            k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops
        }
        loss, metrics = sess.run((spec.loss, update_ops))
        self.assertAllClose(loss, 10.)
        self.assertItemsEqual(metrics.keys(), expected_metrics)

  def test_train(self):
    with tf.Graph().as_default():
      expected_train_result = b'my_train_op'

      def _train_op_fn(loss):
        with tf.control_dependencies((tf.compat.v1.assert_near(
            tf.cast(loss, dtype=tf.float32), 16., name='assert_loss'),)):
          return tf.constant(expected_train_result)

      head1 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), train_op_fn=_train_op_fn, name='head1')
      head2 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), train_op_fn=_train_op_fn, name='head2')
      multi_head = ranking_head.create_multi_ranking_head([head1, head2],
                                                          [1.0, 2.0])

      logits = {
          'head1': tf.convert_to_tensor(value=[[1., 3.], [1., 2.]]),
          'head2': tf.convert_to_tensor(value=[[2., 3.], [2., 2.]]),
      }
      labels = {
          'head1': tf.convert_to_tensor(value=[[0., 1.], [0., 2.]]),
          'head2': tf.convert_to_tensor(value=[[0., 1.], [0., 2.]]),
      }
      # Create estimator spec.
      spec = multi_head.create_estimator_spec(
          features={},
          mode=tf.estimator.ModeKeys.TRAIN,
          logits=logits,
          labels=labels)

      # Assert spec contains expected tensors.
      self.assertIsNotNone(spec.loss)
      self.assertEqual(spec.eval_metric_ops, {})
      self.assertIsNotNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)

      # Assert predictions, loss, and train_op.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        loss, train_result = sess.run((spec.loss, spec.train_op))
        self.assertAllClose(loss, 16.)
        self.assertEqual(expected_train_result, train_result)

  def test_merge_loss(self):
    """Tests for merging losses from multi-head and regularization loss."""
    with tf.Graph().as_default():
      head1 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), name='head1')
      head2 = ranking_head.create_ranking_head(
          loss_fn=_make_loss_fn(), name='head2')
      multi_head = ranking_head.create_multi_ranking_head([head1, head2],
                                                          [1.0, 2.0])
      logits = {
          'head1': tf.convert_to_tensor(value=[[1., 3.], [1., 2.]]),
          'head2': tf.convert_to_tensor(value=[[2., 3.], [2., 2.]]),
      }
      labels = {
          'head1': tf.convert_to_tensor(value=[[0., 1.], [0., 2.]]),
          'head2': tf.convert_to_tensor(value=[[0., 1.], [0., 2.]]),
      }
      regularization_losses = [1.5, 0.5]
      expected_loss = 1. * 4. + 2. * 6. + 1.5 + 0.5

      # Create loss.
      training_loss = multi_head._merge_loss(
          features={},
          mode=tf.estimator.ModeKeys.TRAIN,
          logits=logits,
          labels=labels,
          regularization_losses=regularization_losses)
      with self.cached_session():
        _initialize_variables(self, tf.compat.v1.train.Scaffold())
        self.assertAllClose(training_loss.eval(), expected_loss)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
