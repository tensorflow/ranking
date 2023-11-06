# Copyright 2024 The TensorFlow Ranking Authors.
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
import os

from absl.testing import parameterized
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.extension import task as task_lib
from tensorflow_ranking.python.keras import model as model_lib
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


class PipelineTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("Simple", False), ("BinaryLabel", True))
  def test_pipeline_with_feature_specs(self, convert_labels_to_binary):
    data_dir = self.create_tempdir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with tf.io.TFRecordWriter(data_file) as writer:
      for _ in range(256):
        writer.write(ELWC.SerializeToString())

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

    train_data_config = task_lib.RankingDataConfig(
        input_path=data_file,
        is_training=True,
        global_batch_size=128,
        list_size=2,
        mask_feature_name=_MASK,
        dataset_fn="tfrecord",
        convert_labels_to_binary=convert_labels_to_binary)
    validation_data_config = task_lib.RankingDataConfig(
        input_path=data_file,
        is_training=False,
        global_batch_size=128,
        list_size=2,
        mask_feature_name=_MASK,
        dataset_fn="tfrecord",
        convert_labels_to_binary=convert_labels_to_binary)

    dnn_scorer = model_lib.DNNScorer(hidden_layer_dims=[16, 8], output_units=1)
    model_builder = model_lib.ModelBuilder(
        input_creator=model_lib.FeatureSpecInputCreator(context_feature_spec,
                                                        example_feature_spec),
        preprocessor=model_lib.PreprocessorWithSpec({}),
        scorer=dnn_scorer,
        mask_feature_name=_MASK,
        name="test_model",
    )

    ranking_task_config = task_lib.RankingTaskConfig(
        train_data=train_data_config,
        validation_data=validation_data_config,
        loss="softmax_loss")
    task = task_lib.RankingTask(
        ranking_task_config,
        model_builder=model_builder,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        label_spec=label_spec)

    model = task.build_model()
    metrics = task.build_metrics()
    train_dataset = task.build_inputs(ranking_task_config.train_data)
    vali_dataset = task.build_inputs(ranking_task_config.validation_data)

    task.initialize(model)
    train_iterator = iter(train_dataset)
    vali_iterator = iter(vali_dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(train_iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(vali_iterator), model, metrics=metrics)


if __name__ == "__main__":
  tf.test.main()
