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

"""Tests for ranking head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.estimator import model_fn

from tensorflow_ranking.python import head as ranking_head
from tensorflow_ranking.python import metrics as metrics_lib


def _initialize_variables(test_case, scaffold):
  """Initializes variables for a session.

  Args:
    test_case: A TensorFlowTestCase object.
    scaffold: A Scaffold object.
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
    logits = ops.convert_to_tensor(logits)
    labels = math_ops.to_float(labels)
    weights = features[
        weights_feature_name] if weights_feature_name is not None else 1.
    loss = math_ops.reduce_sum(logits - labels) * math_ops.reduce_sum(weights)
    return loss

  return _loss_fn


class RankingHeadTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()
    self._default_features_dict = {}
    self._default_signature = (
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    logits = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    weights = [1.] * 3
    self._default_logits = logits
    self._default_labels = labels
    self._default_loss = 9.
    self._default_weights = weights
    self._default_weights_feature_name = 'weights'
    self._default_weighted_loss = 27

  def test_name(self):
    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(), name='fake_head')

    self.assertEqual('fake_head', head.name)

  def test_predict(self):
    head = ranking_head.create_ranking_head(loss_fn=_make_loss_fn())
    logits = [[1., 3.], [1., 2.]]
    spec = head.create_estimator_spec(
        features=self._default_features_dict,
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    # Assert spec contains expected tensors.
    self.assertIsNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNone(spec.train_op)
    self.assertItemsEqual((self._default_signature, 'regression', 'predict'),
                          spec.export_outputs.keys())

    # Assert predictions.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits, predictions)
      self.assertAllClose(
          logits, sess.run(spec.export_outputs[self._default_signature].value))

  def test_eval(self):
    metric_fns = {
        'metric/precision@1':
            metrics_lib.make_ranking_metric_fn(
                metrics_lib.RankingMetricKey.PRECISION, topn=1),
    }
    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(), eval_metric_fns=metric_fns)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=self._default_features_dict,
        mode=model_fn.ModeKeys.EVAL,
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
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(self._default_loss, loss)
      self.assertItemsEqual(expected_metrics, metrics.keys())

  def test_train_create_loss(self):
    head = ranking_head.create_ranking_head(loss_fn=_make_loss_fn())
    # Create loss.
    training_loss = head.create_loss(
        features=self._default_features_dict,
        mode=model_fn.ModeKeys.TRAIN,
        logits=self._default_logits,
        labels=self._default_labels)[0]
    with self.cached_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(self._default_loss, training_loss.eval())

  def test_train(self):
    expected_train_result = b'my_train_op'

    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_near(
          math_ops.to_float(self._default_loss),
          math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(), train_op_fn=_train_op_fn)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=self._default_features_dict,
        mode=model_fn.ModeKeys.TRAIN,
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

  def test_train_with_optimizer(self):
    expected_train_result = b'my_train_op'
    expected_loss = self._default_loss

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        with ops.control_dependencies((check_ops.assert_equal(
            math_ops.to_float(expected_loss),
            math_ops.to_float(loss),
            name='assert_loss'),)):
          return constant_op.constant(expected_train_result)

    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(), optimizer=_Optimizer())

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=self._default_features_dict,
        mode=model_fn.ModeKeys.TRAIN,
        logits=self._default_logits,
        labels=self._default_labels)

    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)

  def test_train_with_regularization_losses(self):
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.

    expected_train_result = b'my_train_op'
    expected_loss = expected_regularization_loss + self._default_loss

    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss),
          math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(), train_op_fn=_train_op_fn)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=self._default_features_dict,
        mode=model_fn.ModeKeys.TRAIN,
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
    weights_feature_name = self._default_weights_feature_name

    def _train_op_fn(loss):
      return loss

    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(weights_feature_name), train_op_fn=_train_op_fn)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={weights_feature_name: self._default_weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=self._default_logits,
        labels=self._default_labels)

    # Assert predictions, loss, and train_op.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(self._default_weighted_loss, loss)
      self.assertAllClose(self._default_weighted_loss, train_result)

  def test_multi_dim_weighted_eval(self):
    weights_feature_name = self._default_weights_feature_name
    metric_fns = {
        'metric/precision@1':
            metrics_lib.make_ranking_metric_fn(
                metrics_lib.RankingMetricKey.PRECISION, topn=1),
    }
    head = ranking_head.create_ranking_head(
        loss_fn=_make_loss_fn(weights_feature_name), eval_metric_fns=metric_fns)

    weights = self._default_weights

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={weights_feature_name: weights},
        mode=model_fn.ModeKeys.EVAL,
        logits=self._default_logits,
        labels=self._default_labels)

    expected_metrics = [
        'labels_mean',
        'logits_mean',
        'metric/precision@1',
    ]

    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, metrics = sess.run((spec.loss, update_ops))
      self.assertAllClose(self._default_weighted_loss, loss)
      self.assertItemsEqual(expected_metrics, metrics.keys())


if __name__ == '__main__':
  test.main()
