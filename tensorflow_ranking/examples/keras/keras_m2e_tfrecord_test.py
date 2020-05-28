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

"""Tests for keras_m2e_tfrecord.py."""

import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.examples.keras import keras_m2e_tfrecord
from tensorflow_serving.apis import input_pb2

FLAGS = flags.FLAGS

ELWC = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "query_tokens"
          value { bytes_list { value: ["this", "is", "a", "relevant", "question"] } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "document_tokens"
          value { bytes_list { value: ["this", "is", "a", "relevant", "answer"] } }
        }
        feature {
          key: "relevance"
          value { int64_list { value: 1 } }
        }
        feature {
          key: "doc_weight"
          value { float_list { value: 0.5 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "document_tokens"
          value { bytes_list { value: ["irrelevant", "data"] } }
        }
        feature {
          key: "relevance"
          value { int64_list { value: 0 } }
        }
        feature {
          key: "doc_weight"
          value { float_list { value: 2.0 } }
        }
      }
    }""", input_pb2.ExampleListWithContext())


class KerasM2ETFRecordTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(KerasM2ETFRecordTest, self).setUp()
    data_dir = tf.compat.v1.test.get_temp_dir()
    self._data_file = os.path.join(data_dir, "elwc.tfrecord")
    if tf.io.gfile.exists(self._data_file):
      tf.io.gfile.remove(self._data_file)

    with tf.io.TFRecordWriter(self._data_file) as writer:
      for elwc in [ELWC] * 10:
        writer.write(elwc.SerializeToString())

    self._model_dir = os.path.join(data_dir, "model")

  def tearDown(self):
    super(KerasM2ETFRecordTest, self).tearDown()
    if self._model_dir and tf.io.gfile.exists(self._model_dir):
      tf.io.gfile.rmtree(self._model_dir)
    self._model_dir = None

  @parameterized.named_parameters(("no_weights_feature", None),
                                  ("with_weights_feature", "doc_weight"))
  def test_train_and_eval(self, weights_feature_name):

    with flagsaver.flagsaver(
        train_path=self._data_file,
        eval_path=self._data_file,
        data_format="example_list_with_context",
        model_dir=self._model_dir,
        num_train_steps=10,
        weights_feature_name=weights_feature_name):
      keras_m2e_tfrecord.train_and_eval()


if __name__ == "__main__":
  tf.test.main()
