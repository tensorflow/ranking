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

"""Tests for tf_ranking_tfrecord.py."""

import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.examples import tf_ranking_tfrecord
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


class TfRankingTFRecordTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("enable_listwise_inference", True),
                                  ("disable_listwise_inference", False))
  def test_train_and_eval(self, listwise_inference):
    data_dir = tf.compat.v1.test.get_temp_dir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with tf.io.TFRecordWriter(data_file) as writer:
      for elwc in [ELWC] * 10:
        writer.write(elwc.SerializeToString())

    model_dir = os.path.join(data_dir, "model")

    with flagsaver.flagsaver(
        train_path=data_file,
        eval_path=data_file,
        data_format="example_list_with_context",
        model_dir=model_dir,
        num_train_steps=10,
        listwise_inference=listwise_inference,
        group_size=1,
        weights_feature_name="doc_weight"):
      tf_ranking_tfrecord.train_and_eval()

    if tf.io.gfile.exists(model_dir):
      tf.io.gfile.rmtree(model_dir)

  def test_train_and_eval_with_document_interactions(self):
    data_dir = tf.compat.v1.test.get_temp_dir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with tf.io.TFRecordWriter(data_file) as writer:
      for elwc in [ELWC] * 10:
        writer.write(elwc.SerializeToString())

    model_dir = os.path.join(data_dir, "model")

    with flagsaver.flagsaver(
        train_path=data_file,
        eval_path=data_file,
        data_format="example_list_with_context",
        model_dir=model_dir,
        num_train_steps=10,
        listwise_inference=True,
        use_document_interactions=True,
        group_size=1,
        weights_feature_name="doc_weight"):
      tf_ranking_tfrecord.train_and_eval()

    if tf.io.gfile.exists(model_dir):
      tf.io.gfile.rmtree(model_dir)


if __name__ == "__main__":
  tf.test.main()
