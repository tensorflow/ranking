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

"""Tests for keras_dnn_tfrecord.py."""

import os

from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.examples.keras import keras_dnn_tfrecord
from tensorflow_serving.apis import input_pb2

ELWC = text_format.Parse(
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

EXAMPLE_PROTO_1 = text_format.Parse(
    """
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
    """, tf.train.Example())

EXAMPLE_PROTO_2 = text_format.Parse(
    """
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
    """, tf.train.Example())


class KerasDNNUnitTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Mirrored", "MirroredStrategy"),
      ("MultiWorker", "MultiWorkerMirroredStrategy"))
  def test_train_and_eval(self, strategy):
    data_dir = self.create_tempdir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")

    with tf.io.TFRecordWriter(data_file) as writer:
      for _ in range(256):
        writer.write(ELWC.SerializeToString())

    model_dir = os.path.join(data_dir, "model")

    with flagsaver.flagsaver(
        strategy=strategy,
        train_input_pattern=data_file,
        valid_input_pattern=data_file,
        model_dir=model_dir,
        num_features=3,
        num_train_steps=10,
        num_epochs=2,
        list_size=2,
        train_batch_size=128,
        valid_batch_size=128,
        hidden_layer_dims="16,8",
        loss="softmax_loss",
        export_best_model=True):
      keras_dnn_tfrecord.train_and_eval()

      latest_model_path = os.path.join(model_dir, "export/latest_model")
      self.assertTrue(tf.saved_model.contains_saved_model(latest_model_path))
      self.assertIsInstance(
          tf.keras.models.load_model(latest_model_path), tf.keras.Model)

      latest_model = tf.compat.v2.saved_model.load(export_dir=latest_model_path)
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

      best_model_path = os.path.join(model_dir, "export/best_model_by_metric")
      self.assertTrue(tf.saved_model.contains_saved_model(best_model_path))

      best_model = tf.compat.v2.saved_model.load(export_dir=best_model_path)
      listwise_predictor = best_model.signatures[
          tf.saved_model.PREDICT_METHOD_NAME]
      best_listwise_logits = listwise_predictor(
          tf.convert_to_tensor([ELWC.SerializeToString()] *
                               2))[tf.saved_model.PREDICT_OUTPUTS]
      self.assertAllEqual([2, 2], best_listwise_logits.get_shape().as_list())

      pointwise_predictor = best_model.signatures[
          tf.saved_model.REGRESS_METHOD_NAME]
      pointwise_logits = pointwise_predictor(
          tf.convert_to_tensor([
              EXAMPLE_PROTO_1.SerializeToString(),
              EXAMPLE_PROTO_2.SerializeToString()
          ]))[tf.saved_model.REGRESS_OUTPUTS]
      self.assertAllEqual([2], pointwise_logits.get_shape().as_list())

      self.assertAllClose(pointwise_logits, best_listwise_logits[0])


if __name__ == "__main__":
  tf.test.main()
