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

"""Tests for estimator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.python import data
from tensorflow_ranking.python import estimator as tfr_estimator
from tensorflow_ranking.python import feature as feature_lib
from tensorflow_serving.apis import input_pb2

ELWC_PROTO = text_format.Parse(
    """
    context {
    }
    examples {
      features {
        feature {
          key: "custom_features_1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "custom_features_2"
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
          key: "custom_features_1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "custom_features_3"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

_LABEL_FEATURE = "utility"

_PADDING_LABEL = -1

# Prepares model directory, and train and eval data.
DATA_DIR = tf.compat.v1.test.get_temp_dir()

DATA_FILE = os.path.join(DATA_DIR, "test_elwc.tfrecord")


def context_feature_column():
  return {}


def example_feature_columns():
  """Returns the feature columns."""
  feature_names = ["custom_features_%d" % (i + 1) for i in range(0, 3)]
  return {
      name:
      tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
      for name in feature_names
  }


def _inner_input_fn():
  context_feature_spec = tf.feature_column.make_parse_example_spec(
      list(context_feature_column().values()))
  label_column = tf.feature_column.numeric_column(
      _LABEL_FEATURE, default_value=_PADDING_LABEL)
  example_feature_spec = tf.feature_column.make_parse_example_spec(
      list(example_feature_columns().values()) + [label_column])
  dataset = data.build_ranking_dataset(
      file_pattern=DATA_FILE,
      data_format=data.ELWC,
      batch_size=10,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      list_size=2,
      reader=tf.data.TFRecordDataset)
  features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
  return features, label


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


def _multiply_by_two_transform_fn(features, mode):
  for feature, tensor in six.iteritems(features):
    features[feature] = 2 * tensor

  context, example = feature_lib.encode_listwise_features(
      features=features,
      context_feature_columns=_context_feature_columns(),
      example_feature_columns=_example_feature_columns(),
      mode=mode)
  return context, example


def _get_hparams():
  hparams = dict(
      train_input_pattern="",
      eval_input_pattern="",
      learning_rate=0.01,
      train_batch_size=8,
      eval_batch_size=8,
      checkpoint_secs=120,
      num_checkpoints=100,
      num_train_steps=10000,
      num_eval_steps=100,
      loss="softmax_loss",
      list_size=10,
      listwise_inference=False,
      convert_labels_to_binary=False,
      model_dir=None)
  return hparams


class EstimatorBuilderTest(tf.test.TestCase):

  def _create_default_estimator(self, hparams=None):
    return tfr_estimator.EstimatorBuilder(
        _context_feature_columns(),
        _example_feature_columns(),
        _scoring_function,
        hparams=(hparams or _get_hparams()))

  def test_validate_hparams(self):
    hparams = {"allowed_key": "allowed_value"}
    tfr_estimator._validate_hparams(hparams, [], [])
    tfr_estimator._validate_hparams(hparams, [],
                                    ["allowed_key", "not_set_allowed"])
    with self.assertRaises(ValueError):
      tfr_estimator._validate_hparams(hparams, ["required"])

  def test_create_estimator_with_misspecified_args(self):
    hparams = _get_hparams()
    with self.assertRaises(ValueError):
      _ = tfr_estimator.EstimatorBuilder(
          _context_feature_columns,
          None,  # `document_feature_columns` is None.
          _scoring_function,
          hparams=hparams)

    with self.assertRaises(ValueError):
      _ = tfr_estimator.EstimatorBuilder(
          _context_feature_columns,
          _example_feature_columns,
          None,  # `scoring_function` is None.
          hparams=hparams)

    # Either the optimizer or the hparams["learning_rate"] should be specified.
    del hparams["learning_rate"]
    with self.assertRaises(ValueError):
      _ = tfr_estimator.EstimatorBuilder(
          _context_feature_columns,
          _example_feature_columns,
          _scoring_function,
          optimizer=None,
          hparams=hparams)

    # Passing an optimizer (no hparams["learning_rate"]) will slience the error.
    pip = tfr_estimator.EstimatorBuilder(
        _context_feature_columns,
        _example_feature_columns,
        _scoring_function,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01),
        hparams=_get_hparams())
    self.assertIsInstance(pip, tfr_estimator.EstimatorBuilder)

    # Adding "learning_rate" to hparams (no optimizer) also silences the errors.
    hparams.update(learning_rate=0.01)
    pip = tfr_estimator.EstimatorBuilder(
        _context_feature_columns,
        _example_feature_columns,
        _scoring_function,
        optimizer=None,
        hparams=_get_hparams())
    self.assertIsInstance(pip, tfr_estimator.EstimatorBuilder)

  def test_transform_fn_train(self):
    estimator_with_default_transform_fn = self._create_default_estimator()

    # The below tests the `transform_fn` in the TRAIN mode. In this mode, the
    # `_transform_fn` invokes the `encode_listwise_features()`, which requires
    # 3D example features and 2D context features.
    context, example = estimator_with_default_transform_fn._transform_fn(
        {
            "f1": tf.ones([10, 10, 1], dtype=tf.float32),
            "f2": tf.ones([10, 10, 1], dtype=tf.float32) * 2.0,
            "f3": tf.ones([10, 10, 1], dtype=tf.float32) * 3.0,
            "c1": tf.ones([10, 1], dtype=tf.float32),
            "c2": tf.ones([10, 1], dtype=tf.float32) * 2.0,
        }, tf.estimator.ModeKeys.TRAIN)
    # `c1` is the only context feature defined in `_context_feature_columns()`.
    self.assertCountEqual(context.keys(), ["c1"])

    # `f1`, `f2`, `f3` are all defined in the `_example_feature_columns()`.
    self.assertCountEqual(example.keys(), ["f1", "f2", "f3"])

    # Validates the `context` and `example` features are transformed correctly.
    self.assertAllEqual(tf.ones(shape=[10, 1]), context["c1"])
    self.assertAllEqual(tf.ones(shape=[10, 10, 1]), example["f1"])

  def test_transform_fn_predict_pointwise_inference(self):
    estimator_with_default_transform_fn = self._create_default_estimator()

    # The below tests `transform_fn` in PREDICT mode with pointwise inference.
    # In this mode, `_transform_fn` invokes the `encode_pointwise_features()`,
    # which requires 2D example features and 2D context features.
    context, example = estimator_with_default_transform_fn._transform_fn(
        {
            "f1": tf.ones([10, 1], dtype=tf.float32),
            "f2": tf.ones([10, 1], dtype=tf.float32) * 2.0,
            "f3": tf.ones([10, 1], dtype=tf.float32) * 3.0,
            "c1": tf.ones([10, 1], dtype=tf.float32),
            "c2": tf.ones([10, 1], dtype=tf.float32) * 2.0,
        }, tf.estimator.ModeKeys.PREDICT)

    # After transformation, we get 2D context tensor and 3D example tensor.
    self.assertAllEqual(tf.ones(shape=[10, 1]), context["c1"])
    self.assertAllEqual(tf.ones(shape=[10, 1, 1]), example["f1"])

  def test_transform_fn_listwise_inference(self):
    hparams = _get_hparams()
    hparams["listwise_inference"] = True
    estimator_with_default_transform_fn = self._create_default_estimator(
        hparams=hparams)

    # The below tests the `transform_fn` in the TRAIN mode. In this mode, the
    # `_transform_fn` invokes the `encode_listwise_features()`, which requires
    # 3D example features and 2D context features.
    context, example = estimator_with_default_transform_fn._transform_fn(
        {
            "f1": tf.ones([10, 10, 1], dtype=tf.float32),
            "f2": tf.ones([10, 10, 1], dtype=tf.float32) * 2.0,
            "f3": tf.ones([10, 10, 1], dtype=tf.float32) * 3.0,
            "c1": tf.ones([10, 1], dtype=tf.float32),
            "c2": tf.ones([10, 1], dtype=tf.float32) * 2.0,
        }, tf.estimator.ModeKeys.PREDICT)
    # `c1` is the only context feature defined in `_context_feature_columns()`.
    self.assertCountEqual(context.keys(), ["c1"])

    # `f1`, `f2`, `f3` are all defined in the `_example_feature_columns()`.
    self.assertCountEqual(example.keys(), ["f1", "f2", "f3"])

    # Validates the `context` and `example` features are transformed correctly.
    self.assertAllEqual(tf.ones(shape=[10, 1]), context["c1"])
    self.assertAllEqual(tf.ones(shape=[10, 10, 1]), example["f1"])

  def test_custom_transform_fn(self):
    estimator_with_customized_transform_fn = tfr_estimator.EstimatorBuilder(
        _context_feature_columns(),
        _example_feature_columns(),
        _scoring_function,
        transform_function=_multiply_by_two_transform_fn,
        hparams=_get_hparams())

    context, example = estimator_with_customized_transform_fn._transform_fn(
        {
            "f1": tf.ones([10, 10, 1], dtype=tf.float32),
            "f2": tf.ones([10, 10, 1], dtype=tf.float32) * 2.0,
            "f3": tf.ones([10, 10, 1], dtype=tf.float32) * 3.0,
            "c1": tf.ones([10, 1], dtype=tf.float32),
            "c2": tf.ones([10, 1], dtype=tf.float32) * 2.0,
        }, tf.estimator.ModeKeys.TRAIN)

    self.assertCountEqual(context.keys(), ["c1"])
    self.assertCountEqual(example.keys(), ["f1", "f2", "f3"])
    # By adopting `_multiply_by_two_transform_fn`, the `context` and `example`
    # tensors will be both multiplied by 2.
    self.assertAllEqual(2 * tf.ones(shape=[10, 1]), context["c1"])
    self.assertAllEqual(2 * tf.ones(shape=[10, 10, 1]), example["f1"])

  def test_group_score_fn(self):
    estimator = self._create_default_estimator()
    logits = estimator._group_score_fn(
        {"c1": tf.ones([10, 1], dtype=tf.float32)},
        {"f1": tf.ones([10, 1, 1], dtype=tf.float32)},
        tf.estimator.ModeKeys.TRAIN, None, None)

    self.assertAllEqual(logits, tf.ones([10, 1], dtype=tf.float32))

  def test_eval_metric_fns(self):
    estimator = self._create_default_estimator()
    self.assertCountEqual(estimator._eval_metric_fns().keys(), [
        "metric/mrr", "metric/mrr_10", "metric/ndcg", "metric/ndcg_10",
        "metric/ndcg_5", "metric/softmax_loss"
    ])

    # Metric weights feature name is set.
    hparams = _get_hparams()
    hparams.update({"metric_weights_feature_name": "a_weight_feature"})
    estimator = self._create_default_estimator(hparams=hparams)
    self.assertCountEqual(estimator._eval_metric_fns().keys(), [
        "metric/mrr", "metric/mrr_10", "metric/ndcg", "metric/ndcg_10",
        "metric/ndcg_5", "metric/weighted_mrr", "metric/weighted_mrr_10",
        "metric/weighted_ndcg", "metric/weighted_ndcg_10",
        "metric/weighted_ndcg_5", "metric/softmax_loss",
        "metric/weighted_softmax_loss"
    ])

  def test_optimizer(self):
    estimator_with_default_optimizer = self._create_default_estimator()
    self.assertIsInstance(estimator_with_default_optimizer._optimizer,
                          tf.compat.v1.train.AdagradOptimizer)

    estimator_with_adam_optimizer = tfr_estimator.EstimatorBuilder(
        _context_feature_columns(),
        _example_feature_columns(),
        _scoring_function,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01),
        hparams=_get_hparams())
    self.assertIsInstance(estimator_with_adam_optimizer._optimizer,
                          tf.compat.v1.train.AdamOptimizer)


class DNNEstimatorTest(tf.test.TestCase):

  def test_experiment(self):
    serialized_elwc_list = [
        ELWC_PROTO.SerializeToString(),
    ] * 10

    if tf.io.gfile.exists(DATA_FILE):
      tf.io.gfile.remove(DATA_FILE)
    with tf.io.TFRecordWriter(DATA_FILE) as writer:
      for serialized_elwc in serialized_elwc_list:
        writer.write(serialized_elwc)

    estimator = tfr_estimator.make_dnn_ranking_estimator(
        example_feature_columns=example_feature_columns(),
        hidden_units=["2", "2"],
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        learning_rate=0.05,
        loss="softmax_loss",
        use_batch_norm=False,
        model_dir=None)
    train_spec = tf.estimator.TrainSpec(input_fn=_inner_input_fn, max_steps=1)
    eval_spec = tf.estimator.EvalSpec(
        name="eval", input_fn=_inner_input_fn, steps=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


class GAMEstimatorTest(tf.test.TestCase):

  def test_experiment(self):
    serialized_elwc_list = [
        ELWC_PROTO.SerializeToString(),
    ] * 10

    if tf.io.gfile.exists(DATA_FILE):
      tf.io.gfile.remove(DATA_FILE)
    with tf.io.TFRecordWriter(DATA_FILE) as writer:
      for serialized_elwc in serialized_elwc_list:
        writer.write(serialized_elwc)

    estimator = tfr_estimator.make_gam_ranking_estimator(
        example_feature_columns=example_feature_columns(),
        example_hidden_units=["2", "2"],
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        learning_rate=0.05,
        loss="softmax_loss",
        use_batch_norm=False,
        model_dir=None)
    train_spec = tf.estimator.TrainSpec(input_fn=_inner_input_fn, max_steps=1)
    eval_spec = tf.estimator.EvalSpec(
        name="eval", input_fn=_inner_input_fn, steps=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
  tf.test.main()
