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

"""Tests for pipeline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_ranking as tfr

from google.protobuf import text_format
from tensorflow_ranking.extension import pipeline
from tensorflow_serving.apis import input_pb2

ELWC = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "c1"
          value { float_list { value: 1.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "f1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "f2"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "utility"
          value { float_list { value: 0.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "f1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "f3"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())


def _example_feature_columns():
  return {
      name:
      tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
      for name in ["f1", "f2", "f3"]
  }


def _context_feature_columns():
  return {
      name:
      tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
      for name in ["c1"]
  }


def _scoring_function(context_features, example_features, mode):
  del context_features
  del mode
  batch_size = tf.shape(input=example_features["f1"])[0]
  return tf.ones([batch_size, 1], dtype=tf.float32)


def _write_tfrecord_files(path):
  elwc_list = [ELWC.SerializeToString()] * 10
  if tf.io.gfile.exists(path):
    tf.io.gfile.remove(path)

  with tf.io.TFRecordWriter(path) as writer:
    for elwc in elwc_list:
      writer.write(elwc)


def _make_hparams(train_input_pattern,
                  eval_input_pattern,
                  model_dir,
                  list_size=5,
                  num_train_steps=1,
                  num_eval_steps=1,
                  checkpoint_secs=1,
                  num_checkpoints=2,
                  listwise_inference=False):
  return dict(
      train_input_pattern=train_input_pattern,
      eval_input_pattern=eval_input_pattern,
      learning_rate=0.01,
      train_batch_size=8,
      eval_batch_size=8,
      checkpoint_secs=checkpoint_secs,
      num_checkpoints=num_checkpoints,
      num_train_steps=num_train_steps,
      num_eval_steps=num_eval_steps,
      loss="softmax_loss",
      list_size=list_size,
      listwise_inference=listwise_inference,
      convert_labels_to_binary=False,
      model_dir=model_dir)


class RankingPipelineTest(tf.test.TestCase):

  def setUp(self):
    super(RankingPipelineTest, self).setUp()
    tf.compat.v1.reset_default_graph()

    # Prepares model directory, and train and eval data.
    self._model_dir = tf.compat.v1.test.get_temp_dir() + "/ranking_pipeline/"
    tf.io.gfile.makedirs(self._model_dir)
    self._data_file = os.path.join(self._model_dir, "elwc.tfrecord")
    _write_tfrecord_files(self._data_file)

  def tearDown(self):
    super(RankingPipelineTest, self).tearDown()
    if self._model_dir:
      tf.io.gfile.rmtree(self._model_dir)
    self._model_dir = None

  def _create_pipeline(self):
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=self._model_dir,
        list_size=5)
    estimator = tfr.estimator.EstimatorBuilder(
        _context_feature_columns(),
        _example_feature_columns(),
        _scoring_function,
        hparams=hparams).make_estimator()
    return pipeline.RankingPipeline(
        _context_feature_columns(),
        _example_feature_columns(),
        hparams=hparams,
        estimator=estimator,
        label_feature_name="utility",
        label_feature_type=tf.float32)

  def test_make_input_fn(self):
    batch_size = 1

    pip = self._create_pipeline()
    ds = pip._make_input_fn(
        input_pattern=self._data_file,
        batch_size=batch_size,
        list_size=5,
        randomize_input=False)()
    features, labels = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()

    self.assertCountEqual(features.keys(), ["c1", "f1", "f2", "f3"])
    self.assertAllEqual(
        labels,
        tf.tile(tf.constant([[0., 1., -1., -1., -1.]]), [batch_size, 1]))

  def test_estimator(self):
    pip = self._create_pipeline()
    self.assertIsInstance(pip._estimator, tf.estimator.Estimator)

  def test_train_and_eval(self):

    def _trainable_score_fn(context_features, example_features, mode):
      del context_features, mode
      input_layer = tf.ones_like(example_features["f1"])
      return tf.compat.v1.layers.dense(input_layer, units=1)

    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=self._model_dir,
        list_size=None,
        num_train_steps=1,
        num_eval_steps=1)

    estimator = tfr.estimator.EstimatorBuilder(
        _context_feature_columns(),
        _example_feature_columns(),
        _trainable_score_fn,
        hparams=hparams).make_estimator()

    pip = pipeline.RankingPipeline(
        _context_feature_columns(),
        _example_feature_columns(),
        hparams=hparams,
        estimator=estimator,
        label_feature_name="utility",
        label_feature_type=tf.float32)
    pip.train_and_eval()

    hparams["listwise_inference"] = True
    pip_listwise_export = pipeline.RankingPipeline(
        _context_feature_columns(),
        _example_feature_columns(),
        hparams=hparams,
        estimator=estimator,
        label_feature_name="utility",
        label_feature_type=tf.float32)
    pip_listwise_export.train_and_eval()

    # Does not support non-local training.
    with self.assertRaises(ValueError):
      pip.train_and_eval(local_training=False)

  def test_create_pipeline_with_misspecified_args(self):
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=self._model_dir)

    # The `estimator` cannot be None.
    with self.assertRaises(ValueError):
      _ = pipeline.RankingPipeline(
          _context_feature_columns(),
          _example_feature_columns(),
          hparams=hparams,
          estimator=None,
          label_feature_name="utility",
          label_feature_type=tf.float32)

    # A dict is not an `Estimator`.
    estimator = {}
    with self.assertRaises(ValueError):
      _ = pipeline.RankingPipeline(
          _context_feature_columns(),
          _example_feature_columns(),
          hparams=hparams,
          estimator=estimator,
          label_feature_name="utility",
          label_feature_type=tf.float32)


class RankingPipelineClient(object):
  """A simple pipeline for the integration testing on RankingPipeline."""

  def scoring_function(self, context_features, example_features, mode):
    with tf.compat.v1.name_scope("input_layer"):
      input_features = ([
          tf.compat.v1.layers.flatten(context_features[name])
          for name in sorted(_context_feature_columns())
      ] + [
          tf.compat.v1.layers.flatten(example_features[name])
          for name in sorted(_example_feature_columns())
      ])
      input_layer = tf.concat(input_features, 1)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = tf.compat.v1.layers.batch_normalization(
        input_layer, training=is_training)

    hidden_layer_dims = [16, 8]
    for layer_width in hidden_layer_dims:
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
          cur_layer, training=is_training)
      cur_layer = tf.nn.relu(cur_layer)
    return tf.compat.v1.layers.dense(cur_layer, units=1)

  def build_pipeline(self, hparams, size_feature_name=None):
    estimator = tfr.estimator.EstimatorBuilder(
        context_feature_columns=_context_feature_columns(),
        example_feature_columns=_example_feature_columns(),
        scoring_function=self.scoring_function,
        hparams=hparams).make_estimator()
    return pipeline.RankingPipeline(
        context_feature_columns=_context_feature_columns(),
        example_feature_columns=_example_feature_columns(),
        hparams=hparams,
        estimator=estimator,
        label_feature_name="utility",
        label_feature_type=tf.float32,
        size_feature_name=size_feature_name)

  def build_best_exporter_pipeline(self, hparams):
    estimator = tfr.estimator.EstimatorBuilder(
        context_feature_columns=_context_feature_columns(),
        example_feature_columns=_example_feature_columns(),
        scoring_function=self.scoring_function,
        hparams=hparams).make_estimator()
    return pipeline.RankingPipeline(
        context_feature_columns=_context_feature_columns(),
        example_feature_columns=_example_feature_columns(),
        hparams=hparams,
        estimator=estimator,
        label_feature_name="utility",
        label_feature_type=tf.float32,
        best_exporter_metric="metric/ndcg_5")


class RankingPipelineIntegrationTest(tf.test.TestCase):

  def setUp(self):
    super(RankingPipelineIntegrationTest, self).setUp()
    tf.compat.v1.reset_default_graph()

    # Prepares model directory, and train and eval data.
    self._model_dir = tf.compat.v1.test.get_temp_dir() + "/model/"
    tf.io.gfile.makedirs(self._model_dir)
    self._data_file = os.path.join(self._model_dir, "elwc.tfrecord")
    _write_tfrecord_files(self._data_file)

  def tearDown(self):
    super(RankingPipelineIntegrationTest, self).tearDown()
    if self._model_dir:
      tf.io.gfile.rmtree(self._model_dir)
    self._model_dir = None

  def test_pipeline(self):
    model_dir = self._model_dir + "pipeline/"
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=model_dir,
        list_size=5,
        num_train_steps=3,
        num_eval_steps=3,
        checkpoint_secs=1,
        num_checkpoints=2)

    pip = RankingPipelineClient().build_pipeline(hparams)
    pip.train_and_eval(local_training=True)

    required_patterns = [
        r"model\.ckpt\-\d\.data\-00000\-of\-00001",
        r"model\.ckpt\-\d\.index",
        r"model\.ckpt\-\d\.meta",
    ]
    output_files = tf.io.gfile.listdir(model_dir)
    for pattern in required_patterns:
      self.assertRegex(",".join(output_files), pattern)

  def test_pipeline_with_best_exporter(self):
    model_dir = self._model_dir + "pipeline-exporter/"
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=model_dir,
        list_size=5,
        num_train_steps=3,
        num_eval_steps=3,
        checkpoint_secs=1,
        num_checkpoints=2)

    pip = RankingPipelineClient().build_best_exporter_pipeline(hparams)
    pip.train_and_eval(local_training=True)
    self.assertTrue(tf.io.gfile.exists(model_dir + "export/latest_model"))
    self.assertTrue(
        tf.io.gfile.exists(model_dir + "export/best_model_by_metric"))

  def test_pipeline_with_listwise_inference(self):
    model_dir = self._model_dir + "pipeline-elwc/"
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=model_dir,
        list_size=5,
        num_train_steps=3,
        num_eval_steps=3,
        checkpoint_secs=1,
        num_checkpoints=2,
        listwise_inference=True)

    pip = RankingPipelineClient().build_pipeline(hparams)
    pip.train_and_eval(local_training=True)

    required_patterns = [
        r"model\.ckpt\-\d\.data\-00000\-of\-00001",
        r"model\.ckpt\-\d\.index",
        r"model\.ckpt\-\d\.meta",
    ]
    output_files = tf.io.gfile.listdir(model_dir)
    for pattern in required_patterns:
      self.assertRegex(",".join(output_files), pattern)

  def test_pipeline_with_size_feature_name(self):
    model_dir = self._model_dir + "pipeline-size-feature/"
    hparams = _make_hparams(
        train_input_pattern=self._data_file,
        eval_input_pattern=self._data_file,
        model_dir=model_dir,
        list_size=5,
        num_train_steps=3,
        num_eval_steps=3,
        checkpoint_secs=1,
        num_checkpoints=2)

    pip = RankingPipelineClient().build_pipeline(
        hparams, size_feature_name="example_list_size")
    pip.train_and_eval(local_training=True)

    required_patterns = [
        r"model\.ckpt\-\d\.data\-00000\-of\-00001",
        r"model\.ckpt\-\d\.index",
        r"model\.ckpt\-\d\.meta",
    ]
    output_files = tf.io.gfile.listdir(model_dir)
    for pattern in required_patterns:
      self.assertRegex(",".join(output_files), pattern)


if __name__ == "__main__":
  tf.test.main()
