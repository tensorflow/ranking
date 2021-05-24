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

# Lint as: python3
"""Tests for pipeline.py."""
import os
from typing import Dict
from absl.testing import parameterized

import tensorflow as tf

from tensorflow_ranking.python.keras import model as model_lib
from tensorflow_ranking.python.keras import pipeline

from google.protobuf import text_format
from tensorflow_serving.apis import input_pb2

_MASK = "__list_mask__"
_LABEL_FEATURE = "utility"
_PADDING_LABEL = -1

ELWC = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "cf_1"
          value { float_list { value: 1.0 } }
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

EXAMPLE_PROTO_1 = text_format.Parse(
    """
    features {
      feature {
        key: "cf_1"
        value { float_list { value: 1.0 } }
      }
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
    """, tf.train.Example())

EXAMPLE_PROTO_2 = text_format.Parse(
    """
    features {
      feature {
        key: "cf_1"
        value { float_list { value: 1.0 } }
      }
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
    """, tf.train.Example())


class DummyMultiTaskScorer(model_lib.UnivariateScorer):

  def _score_flattened(
      self,
      context_features: Dict[str, tf.Tensor],
      example_features: Dict[str, tf.Tensor],
  ) -> Dict[str, tf.Tensor]:
    f = next(iter(example_features.values()))
    return {
        "task1": tf.keras.layers.Dense(1)(f),
        "task2": tf.keras.layers.Dense(1)(f),
    }


class PipelineTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Local", None), ("Mirrored", "MirroredStrategy"),
      ("MultiWorker", "MultiWorkerMirroredStrategy"))
  def test_pipeline_with_feature_specs(self, strategy):
    data_dir = self.create_tempdir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with tf.io.TFRecordWriter(data_file) as writer:
      for _ in range(256):
        writer.write(ELWC.SerializeToString())

    model_dir = os.path.join(data_dir, "model")

    dataset_hparams = pipeline.DatasetHparams(
        train_input_pattern=data_file,
        valid_input_pattern=data_file,
        train_batch_size=128,
        valid_batch_size=128,
        list_size=2,
        dataset_reader=tf.data.TFRecordDataset,
        convert_labels_to_binary=False)
    pipeline_hparams = pipeline.PipelineHparams(
        model_dir=model_dir,
        num_epochs=2,
        steps_per_epoch=5,
        validation_steps=2,
        learning_rate=0.01,
        loss="softmax_loss",
        export_best_model=True,
        automatic_reduce_lr=True,
        strategy=strategy)

    context_feature_spec = {
        "cf_1":
            tf.io.FixedLenFeature(
                shape=(1,), dtype=tf.float32, default_value=0.0),
    }
    example_feature_spec = {
        "custom_features_{}".format(i + 1):
        tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0.0)
        for i in range(3)
    }
    label_spec = (_LABEL_FEATURE,
                  tf.io.FixedLenFeature(
                      shape=(1,),
                      dtype=tf.float32,
                      default_value=_PADDING_LABEL))

    dnn_scorer = model_lib.DNNScorer(hidden_layer_dims=[16, 8], output_units=1)
    model_builder = model_lib.ModelBuilder(
        input_creator=model_lib.FeatureSpecInputCreator(context_feature_spec,
                                                        example_feature_spec),
        preprocessor=model_lib.PreprocessorWithSpec({}),
        scorer=dnn_scorer,
        mask_feature_name=_MASK,
        name="test_model",
    )

    ranking_pipeline = pipeline.SimplePipeline(
        model_builder,
        dataset_builder=pipeline.SimpleDatasetBuilder(
            context_feature_spec,
            example_feature_spec,
            _MASK,
            label_spec,
            dataset_hparams),
        hparams=pipeline_hparams)

    ranking_pipeline.train_and_validate(verbose=1)

    latest_model_path = os.path.join(model_dir, "export/latest_model")
    self.assertTrue(tf.saved_model.contains_saved_model(latest_model_path))

    latest_model = tf.saved_model.load(export_dir=latest_model_path)
    listwise_predictor = latest_model.signatures[
        tf.saved_model.PREDICT_METHOD_NAME]
    listwise_logits = listwise_predictor(
        tf.convert_to_tensor([ELWC.SerializeToString()] *
                             2))[tf.saved_model.PREDICT_OUTPUTS]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())

    pointwise_predictor = latest_model.signatures[
        tf.saved_model.REGRESS_METHOD_NAME]
    pointwise_logits = pointwise_predictor(
        tf.convert_to_tensor([
            EXAMPLE_PROTO_1.SerializeToString(),
            EXAMPLE_PROTO_2.SerializeToString()
        ]))[tf.saved_model.REGRESS_OUTPUTS]
    self.assertAllEqual([2], pointwise_logits.get_shape().as_list())

    self.assertAllClose(pointwise_logits, listwise_logits[0])

  def test_pipeline_with_multi_task(self):
    data_dir = self.create_tempdir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with tf.io.TFRecordWriter(data_file) as writer:
      for _ in range(256):
        writer.write(ELWC.SerializeToString())

    model_dir = os.path.join(data_dir, "model")

    dataset_hparams = pipeline.DatasetHparams(
        train_input_pattern=data_file,
        valid_input_pattern=data_file,
        train_batch_size=128,
        valid_batch_size=128,
        list_size=2,
        dataset_reader=tf.data.TFRecordDataset,
        convert_labels_to_binary=False)
    pipeline_hparams = pipeline.PipelineHparams(
        model_dir=model_dir,
        num_epochs=2,
        steps_per_epoch=5,
        validation_steps=2,
        learning_rate=0.01,
        loss={
            "task1": "softmax_loss",
            "task2": "pairwise_logistic_loss"
        },
        loss_weights={
            "task1": 1.0,
            "task2": 2.0
        },
        export_best_model=True,
        strategy="MirroredStrategy")

    context_feature_spec = {
        "cf_1":
            tf.io.FixedLenFeature(
                shape=(1,), dtype=tf.float32, default_value=0.0),
    }
    example_feature_spec = {
        "custom_features_{}".format(i + 1):
        tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0.0)
        for i in range(3)
    }
    label_spec = (_LABEL_FEATURE,
                  tf.io.FixedLenFeature(
                      shape=(1,),
                      dtype=tf.float32,
                      default_value=_PADDING_LABEL))
    label_spec = {"task1": label_spec, "task2": label_spec}
    weight_spec = ("weight",
                   tf.io.FixedLenFeature(
                       shape=(1,), dtype=tf.float32, default_value=1.))

    model_builder = model_lib.ModelBuilder(
        input_creator=model_lib.FeatureSpecInputCreator(context_feature_spec,
                                                        example_feature_spec),
        preprocessor=model_lib.PreprocessorWithSpec({}),
        scorer=DummyMultiTaskScorer(),
        mask_feature_name=_MASK,
        name="multi_task_model",
    )

    ranking_pipeline = pipeline.MultiTaskPipeline(
        model_builder,
        dataset_builder=pipeline.MultiLabelDatasetBuilder(
            context_feature_spec,
            example_feature_spec,
            _MASK,
            label_spec,
            dataset_hparams,
            sample_weight_spec=weight_spec),
        hparams=pipeline_hparams)

    ranking_pipeline.train_and_validate(verbose=1)

    latest_model_path = os.path.join(model_dir, "export/latest_model")
    self.assertTrue(tf.saved_model.contains_saved_model(latest_model_path))

    latest_model = tf.saved_model.load(export_dir=latest_model_path)
    listwise_predictor = latest_model.signatures[
        tf.saved_model.PREDICT_METHOD_NAME]
    listwise_logits = listwise_predictor(
        tf.convert_to_tensor([ELWC.SerializeToString()] * 2))["task1"]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())

    pointwise_predictor = latest_model.signatures[
        tf.saved_model.REGRESS_METHOD_NAME]
    pointwise_logits = pointwise_predictor(
        tf.convert_to_tensor([
            EXAMPLE_PROTO_1.SerializeToString(),
            EXAMPLE_PROTO_2.SerializeToString()
        ]))["task1"]
    self.assertAllEqual([2], pointwise_logits.get_shape().as_list())

    self.assertAllClose(pointwise_logits, listwise_logits[0])


if __name__ == "__main__":
  tf.test.main()
