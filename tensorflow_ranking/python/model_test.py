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

"""Tests for model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_ranking.python import feature
from tensorflow_ranking.python import head
from tensorflow_ranking.python import losses
from tensorflow_ranking.python import model

# Names of variables created by the model.
BIAS_NAME = 'group_score/dense/bias'
WEIGHTS_NAME = 'group_score/dense/kernel'
ADAGRAD_BIAS_NAME = 'group_score/dense/bias/Adagrad'
ADAGRAD_WEIGHTS_NAME = 'group_score/dense/kernel/Adagrad'


class UtilsTest(tf.test.TestCase):
  """Tests for util functions."""

  def test_get_params(self):
    params = {
        'num_shuffles_train': 1,
        'num_shuffles_eval': 2,
        'num_shuffles_predict': 3,
    }
    self.assertEqual(model._get_params(tf.estimator.ModeKeys.TRAIN, params), 1)
    self.assertEqual(model._get_params(tf.estimator.ModeKeys.EVAL, params), 2)
    self.assertEqual(
        model._get_params(tf.estimator.ModeKeys.PREDICT, params), 3)

  def test_rolling_window_indices(self):
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        # All valid.
        indices, mask = sess.run(model._rolling_window_indices(3, 2, [3]))
        self.assertAllEqual(indices, [[[0, 1], [1, 2], [2, 0]]])
        self.assertAllEqual(mask, [[True, True, True]])
        # One invalid.
        indices, mask = sess.run(model._rolling_window_indices(3, 2, [2]))
        self.assertAllEqual(indices, [[[0, 1], [1, 0], [0, 1]]])
        self.assertAllEqual(mask, [[True, True, False]])
        # All invalid.
        indices, mask = sess.run(model._rolling_window_indices(3, 2, [0]))
        self.assertAllEqual(indices, [[[0, 0], [0, 0], [0, 0]]])
        self.assertAllEqual(mask, [[False, False, False]])
        # size < rw_size.
        indices, mask = sess.run(model._rolling_window_indices(2, 3, [2]))
        self.assertAllEqual(indices, [[[0, 1, 0], [1, 0, 1]]])
        self.assertAllEqual(mask, [[True, True]])
        # batch_size = 2
        indices, mask = sess.run(model._rolling_window_indices(3, 2, [3, 2]))
        self.assertAllEqual(
            indices, [[[0, 1], [1, 2], [2, 0]], [[0, 1], [1, 0], [0, 1]]])
        self.assertAllEqual(mask, [[True, True, True], [True, True, False]])

  def test_form_group_indices_nd(self):
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(1)
      # batch_size, list_size = 2, 3.
      is_valid = [[True, True, True], [True, True, False]]
      indices, mask = model._form_group_indices_nd(
          is_valid, group_size=2, shuffle=True, seed=87124)

      with tf.compat.v1.Session() as sess:
        indices, mask = sess.run([indices, mask])
        # shape = [2, 3, 2, 2] = [batch_size, num_groups , group_size, 2].
        self.assertAllEqual(
            indices,
            [  # batch_size = 2.
                [  # num_groups = 3.
                    [[0, 0], [0, 1]], [[0, 1], [0, 2]], [[0, 2], [0, 0]]
                ],
                [  # num_groups = 3.
                    [[1, 1], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [1, 0]]
                ]
            ])
        # shape = [2, 3] = [batch_size, num_groups]
        self.assertAllEqual(mask, [[True, True, True], [True, True, False]])

      # Disable shuffling.
      indices, mask = model._form_group_indices_nd(
          is_valid, group_size=2, shuffle=False)

      with tf.compat.v1.Session() as sess:
        indices, mask = sess.run([indices, mask])
        # shape = [2, 3, 2, 2] = [batch_size, num_groups , group_size, 2].
        self.assertAllEqual(
            indices,
            [  # batch_size = 2.
                [  # num_groups = 3.
                    [[0, 0], [0, 1]], [[0, 1], [0, 2]], [[0, 2], [0, 0]]
                ],
                [  # num_groups = 3.
                    [[1, 0], [1, 1]], [[1, 1], [1, 0]], [[1, 0], [1, 1]]
                ]
            ])
        # shape = [2, 3] = [batch_size, num_groups]
        self.assertAllEqual(mask, [[True, True, True], [True, True, False]])


def _save_variables_to_ckpt(model_dir):
  """Save all graph variables in a checkpoint under 'model_dir'."""
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    tf.compat.v1.train.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))


def _group_score_fn(context_features, group_features, mode, params, config):
  """Scoring function of the groupwise ranking model_fn under test."""
  del [mode, params, config]
  batch_size = tf.shape(input=context_features['context'])[0]
  input_layer = tf.concat([
      tf.reshape(context_features['context'], [batch_size, -1]),
      tf.reshape(group_features['age'], [batch_size, -1])
  ],
                          axis=1)
  # Shape is [batch_size, 2].
  # TODO: Convert to tf.keras.layers.Dense, and change *_NAME.
  group_score = tf.compat.v1.layers.dense(input_layer, units=2)
  return group_score


def _train_input_fn():
  """Fake input function that returns a static number of list of pairs."""
  # batch_size = 2, list_size = 3.
  features = {
      # Context features with shape [2, 1]
      'context': tf.constant([[178.], [155.]]),
      # Listwise weight with shape [2, 1]
      'weight': tf.constant([[1.], [1.]]),
      # Example features with shape [2, 3, 1] for 3 examples.
      'age': tf.constant([[[10.], [20.], [20.]], [[50.], [30.], [30.]]])
  }
  # Label with shape [2, 3].
  label = tf.constant([[1., 0., 0.], [1., 0., 0.]])
  return features, label


class GroupwiseRankingModelTest(tf.test.TestCase, parameterized.TestCase):
  """GroupwiseRankingModel tests."""

  def test_update_scatter_gather_indices_groupsize_1(self):
    """Test for group size = 1."""
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(1)
      with tf.compat.v1.Session() as sess:
        ranking_model = model._GroupwiseRankingModel(None, group_size=1)
        ranking_model._update_scatter_gather_indices(
            tf.convert_to_tensor(value=[[True, True, False]]),
            tf.estimator.ModeKeys.TRAIN, None)
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 3, 1, 2])
        self.assertEqual(ranking_model._indices_mask.get_shape().as_list(),
                         [1, 3])
        feature_gather_indices, indices_mask = sess.run([
            ranking_model._feature_gather_indices, ranking_model._indices_mask
        ])
        self.assertAllEqual(feature_gather_indices,
                            [[[[0, 0]], [[0, 1]], [[0, 0]]]])
        self.assertAllEqual(indices_mask, [[True, True, False]])

  def test_update_scatter_gather_indices_predict_no_shuffle(self):
    """Test for group size > 1 and mode = PREDICT."""
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(1)
      with tf.compat.v1.Session() as sess:
        ranking_model = model._GroupwiseRankingModel(None, group_size=2)
        ranking_model._update_scatter_gather_indices(
            tf.convert_to_tensor(value=[[True, True, True]]),
            tf.estimator.ModeKeys.PREDICT, None)
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 3, 2, 2])
        self.assertEqual(ranking_model._indices_mask.get_shape().as_list(),
                         [1, 3])
        feature_gather_indices, indices_mask = sess.run([
            ranking_model._feature_gather_indices, ranking_model._indices_mask
        ])
        self.assertAllEqual(
            feature_gather_indices,
            [[[[0, 0], [0, 1]], [[0, 1], [0, 2]], [[0, 2], [0, 0]]]])
        self.assertAllEqual(indices_mask, [[True, True, True]])

  @parameterized.named_parameters(
      ('mode_train', tf.estimator.ModeKeys.TRAIN),
      ('mode_eval', tf.estimator.ModeKeys.EVAL),
      ('mode_predict', tf.estimator.ModeKeys.PREDICT))
  def test_update_scatter_gather_indices(self, mode):
    """Test for group size > 1."""
    params = {
        'num_shuffles_train': 2,
        'num_shuffles_eval': 2,
        'num_shuffles_predict': 2,
    }
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(2)
      with tf.compat.v1.Session() as sess:
        ranking_model = model._GroupwiseRankingModel(None, group_size=2)
        ranking_model._update_scatter_gather_indices(
            tf.convert_to_tensor(value=[[True, True, False]]), mode, params)
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 6, 2, 2])
        self.assertEqual(ranking_model._indices_mask.get_shape().as_list(),
                         [1, 6])
        feature_gather_indices, indices_mask = sess.run([
            ranking_model._feature_gather_indices, ranking_model._indices_mask
        ])

        self.assertAllEqual(
            feature_gather_indices,
            [[
                [[0, 0], [0, 1]],
                [[0, 1], [0, 0]],
                [[0, 0], [0, 1]],  # shuffle 1.
                [[0, 1], [0, 0]],
                [[0, 0], [0, 1]],
                [[0, 1], [0, 0]],  # shuffle 2.
            ]])
        self.assertAllEqual(indices_mask,
                            [[True, True, False, True, True, False]])

  @parameterized.named_parameters(
      ('mode_train', tf.estimator.ModeKeys.TRAIN),
      ('mode_eval', tf.estimator.ModeKeys.EVAL),
      ('mode_predict', tf.estimator.ModeKeys.PREDICT))
  def test_compute_logits(self, mode):
    group_size = 2
    params = {
        'num_shuffles_train': 2,
        'num_shuffles_eval': 2,
        'num_shuffles_predict': 2,
    }

    def _dummy_score_fn(context_features, group_features, mode, params, config):
      del [mode, params, config]
      # 'context': [batch_size * num_groups, 1]
      # 'example_f1': [batch_size * num_groups, group_size, 1]
      logits = tf.expand_dims(
          context_features['context'], axis=1) + group_features['example_f1']
      logits = tf.reshape(logits, [-1, group_size])
      # Add the shape of the logits to differentiate number of shuffles.
      return logits + tf.cast(tf.shape(input=logits)[0], tf.float32)

    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(1)
      with tf.compat.v1.Session() as sess:
        ranking_model = model._GroupwiseRankingModel(
            _dummy_score_fn,
            group_size=group_size,
            transform_fn=feature.make_identity_transform_fn(['context']),
        )

        # batch_size = 1, list_size = 3, is_valid = [True, True, False]
        features = {
            'context': [[1.]],
            'example_f1': [[[1.], [2.], [3.]]],
        }
        labels = [[1., 0, -1]]
        # No params.
        logits = sess.run(
            ranking_model.compute_logits(features, labels, mode, None, None))
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 3, 2, 2])
        self.assertAllEqual(logits, [[5., 6., 0.]])
        # Trigger params.
        logits = sess.run(
            ranking_model.compute_logits(features, labels, mode, params, None))
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 6, 2, 2])
        self.assertAllEqual(logits, [[8., 9., 0.]])

        # batch_size = 1, list_size = 3, is_valid = [True, True, True]
        features = {
            'context': [[1.]],
            'example_f1': [[[1.], [2.], [0.]]],
        }
        labels = [[1., 0, 1]]
        logits = sess.run(
            ranking_model.compute_logits(features, labels, mode, params, None))
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 6, 2, 2])
        self.assertAllEqual(logits, [[8., 9., 7.]])

  @parameterized.named_parameters(
      ('mode_train', tf.estimator.ModeKeys.TRAIN),
      ('mode_eval', tf.estimator.ModeKeys.EVAL),
      ('mode_predict', tf.estimator.ModeKeys.PREDICT))
  def test_compute_logits_multi_task(self, mode):
    group_size = 2

    def _multi_task_score_fn(context_features, group_features, mode, params,
                             config):
      del [mode, params, config]
      # 'context': [batch_size * num_groups, 1]
      # 'example_f1': [batch_size * num_groups, group_size, 1]
      logits = tf.expand_dims(
          context_features['context'], axis=1) + group_features['example_f1']
      logits = tf.reshape(logits, [-1, group_size])
      # Add the shape of the logits to differentiate number of shuffles.
      return {
          'task1': logits + tf.cast(tf.shape(input=logits)[0], tf.float32),
          'task2': logits + tf.cast(tf.shape(input=logits)[0], tf.float32) + 1,
      }

    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(1)
      with tf.compat.v1.Session() as sess:
        ranking_model = model._GroupwiseRankingModel(
            _multi_task_score_fn,
            group_size=group_size,
            transform_fn=feature.make_identity_transform_fn(['context']),
        )

        # batch_size = 1, list_size = 3, is_valid = [True, True, False]
        features = {
            'context': [[1.]],
            'example_f1': [[[1.], [2.], [3.]]],
        }
        labels = {
            'task1': [[1., 0, -1]],
            'task2': [[0., 1, -1]],
        }
        logits = sess.run(
            ranking_model.compute_logits(features, labels, mode, None, None))
        self.assertEqual(
            ranking_model._feature_gather_indices.get_shape().as_list(),
            [1, 3, 2, 2])
        self.assertAllEqual(logits['task1'], [[5., 6., 0.]])
        self.assertAllEqual(logits['task2'], [[6., 7., 0.]])


class GroupwiseRankingEstimatorTest(tf.test.TestCase):
  """Groupwise RankingEstimator tests."""

  def setUp(self):
    super(GroupwiseRankingEstimatorTest, self).setUp()
    tf.compat.v1.reset_default_graph()
    self._model_dir = tf.compat.v1.test.get_temp_dir()
    tf.io.gfile.makedirs(self._model_dir)
    model_fn = model.make_groupwise_ranking_fn(
        _group_score_fn,
        group_size=2,
        transform_fn=feature.make_identity_transform_fn(['context', 'weight']),
        ranking_head=head.create_ranking_head(
            loss_fn=losses.make_loss_fn(
                losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
                weights_feature_name='weight'),
            optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)))
    self._estimator = tf.estimator.Estimator(model_fn, self._model_dir)

  def tearDown(self):
    if self._model_dir:
      tf.io.gfile.rmtree(self._model_dir)
    self._model_dir = None
    self._estimator = None

  def _assert_checkpoint(self,
                         expected_global_step,
                         expected_weights=None,
                         expected_bias=None):
    """Assert the values and shapes of the variables saved in the checkpoint."""
    shapes = {
        name: shape
        for (name, shape) in tf.train.list_variables(self._model_dir)
    }

    reader = tf.train.load_checkpoint(self._model_dir)

    self.assertEqual([], shapes[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertEqual(expected_global_step,
                     reader.get_tensor(tf.compat.v1.GraphKeys.GLOBAL_STEP))

    self.assertEqual([3, 2], shapes[WEIGHTS_NAME])
    if expected_weights is not None:
      self.assertAllClose(expected_weights, reader.get_tensor(WEIGHTS_NAME))

    self.assertEqual([2], shapes[BIAS_NAME])
    if expected_bias is not None:
      self.assertAllClose(expected_bias, reader.get_tensor(BIAS_NAME))

  def _initialize_checkpoint(self):
    """Initialize the model checkpoint with constant values."""
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(23)
      tf.Variable([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], name=WEIGHTS_NAME)
      tf.Variable([1.0, 1.0], name=BIAS_NAME)
      tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.int64)
      # Adagrad weights should not be zero, otherwise we get NaNs.
      tf.Variable([[1e-12, 1e-12], [1e-12, 1e-12], [1e-12, 1e-12]],
                  name=ADAGRAD_WEIGHTS_NAME)
      tf.Variable([1e-12, 1e-12], name=ADAGRAD_BIAS_NAME)

      _save_variables_to_ckpt(self._model_dir)

  def test_train(self):
    """Train the estimator for one step and check checkpoint."""
    self._estimator.train(input_fn=_train_input_fn, steps=1)
    self._assert_checkpoint(expected_global_step=1)

  def test_train_from_checkpoint(self):
    """Load a static checkpoint and train estimator for 1 step."""
    self._initialize_checkpoint()
    self._estimator.train(input_fn=_train_input_fn, steps=1)
    self._assert_checkpoint(
        expected_global_step=101,
        expected_bias=[1.0, 1.0],
        expected_weights=[[1., 1.], [1.9, 2.1], [3.1, 2.9]])

  def test_eval(self):
    """Load a static checkpoint and test eval function."""
    self._initialize_checkpoint()
    self._assert_checkpoint(
        expected_global_step=100,
        expected_bias=[1.0, 1.0],
        expected_weights=[[1., 1.], [2., 2.], [3., 3.]])

    eval_output = self._estimator.evaluate(input_fn=_train_input_fn, steps=1)
    self.assertAllClose(
        {
            'global_step': 100,
            'loss': 6.75,
            'labels_mean': 0.3333333,
            'logits_mean': 300.833343,
        },
        dict((name, eval_output[name])
             for name in ['global_step', 'loss', 'labels_mean', 'logits_mean']))

  def test_predict(self):
    """Load a static checkpoint and test predict function."""
    self._initialize_checkpoint()
    self._assert_checkpoint(
        expected_global_step=100,
        expected_bias=[1.0, 1.0],
        expected_weights=[[1., 1.], [2., 2.], [3., 3.]])
    # When infer_list_size = 2, inference will have the following groups [0, 1],
    # [1, 0] given group_size = 2.
    features = {
        'context': [[178.], [155.]],
        'age': [[[10.], [20.]], [[50.], [30.]]],
    }
    predictions = self._estimator.predict(input_fn=lambda: (features, None))
    self.assertAllClose([254., 254.], list(next(predictions)))
    self.assertAllClose([356., 356.], list(next(predictions)))

    # When infer_list_size = 4, inference will have the following groups [0, 1],
    # [1, 2], [2, 3], [3, 0] given group_size = 2.
    features = {
        'context': [[178.], [178.], [178.]],
        'age': [[[20.], [10.], [10.], [10.]], [[20.], [10.], [10.], [10.]],
                [[20.], [10.], [10.], [10.]]],
    }
    predictions = self._estimator.predict(input_fn=lambda: (features, None))
    self.assertAllClose([254., 239., 229., 244.], list(next(predictions)))
    self.assertAllClose([254., 239., 229., 244.], list(next(predictions)))
    self.assertAllClose([254., 239., 229., 244.], list(next(predictions)))

    # Evaluation after training.
    self._estimator.train(input_fn=_train_input_fn, steps=1)

    features = {
        'context': [[178.], [155.]],
        'age': [[[10.], [20.]], [[50.], [30.]]],
    }
    predictions = self._estimator.predict(input_fn=lambda: (features, None))
    self.assertAllClose([255., 253.], list(next(predictions)))
    self.assertAllClose([354., 358.], list(next(predictions)))

    features = {
        'context': [[178.], [178.], [178.]],
        'age': [[[20.], [10.], [10.], [10.]], [[20.], [10.], [10.], [10.]],
                [[20.], [10.], [10.], [10.]]],
    }
    predictions = self._estimator.predict(input_fn=lambda: (features, None))
    self.assertAllClose([253., 239.5, 229., 244.5], list(next(predictions)))
    self.assertAllClose([253., 239.5, 229., 244.5], list(next(predictions)))
    self.assertAllClose([253., 239.5, 229., 244.5], list(next(predictions)))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
