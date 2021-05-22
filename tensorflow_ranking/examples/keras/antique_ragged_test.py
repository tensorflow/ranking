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

"""Tests for antique_kpl_din.py."""

import os

from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.examples.keras import antique_ragged
from tensorflow_serving.apis import input_pb2

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
      }
    }""", input_pb2.ExampleListWithContext())

VOCAB = [
    "this", "is", "a", "relevant", "question", "answer", "irrelevant", "data"
]


class ModelTrainAndEvaluateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((False, "export"), (True, "export/latest_model"))
  def test_train_and_eval(self, use_pipeline, saved_model_dir):
    # `create_tempdir` creates an isolated directory for this test, and
    # will be properly cleaned up by the test.
    data_dir = self.create_tempdir()
    data_file = os.path.join(data_dir, "elwc.tfrecord")
    vocab_file = os.path.join(data_dir, "vocab.txt")
    model_dir = os.path.join(data_dir, "model")

    # Save data.
    with tf.io.TFRecordWriter(data_file) as writer:
      for _ in range(10):
        writer.write(ELWC.SerializeToString())

    with tf.io.gfile.GFile(vocab_file, "w") as writer:
      writer.write("\n".join(VOCAB) + "\n")

    with flagsaver.flagsaver(
        train_input_pattern=data_file,
        eval_input_pattern=data_file,
        model_dir=model_dir,
        vocab_file_path=vocab_file,
        train_batch_size=2,
        eval_batch_size=2,
        num_epochs=5,
        num_train_steps=10,
        num_valid_steps=10,
        vocab_size=len(VOCAB),
        use_pipeline=use_pipeline,
        hidden_layer_dims=[16, 8]):

      # Train and eval model.
      tf.random.set_seed(1234)
      antique_ragged.main(0)

      # Check SavedModel is exported correctly.
      saved_model_path = os.path.join(model_dir, saved_model_dir)
      self.assertTrue(tf.saved_model.contains_saved_model(saved_model_path))

    # Check SavedModel can be loaded and called with ragged tensors.
    saved_model = tf.keras.models.load_model(saved_model_path)
    inputs = {
        "query_tokens":
            tf.ragged.constant([["this", "is", "a", "relevant", "question"]],
                               row_splits_dtype=tf.int32),
        "document_tokens":
            tf.ragged.constant([[["this", "is", "a", "relevant", "answer"],
                                 ["irrelevant", "data"]]],
                               row_splits_dtype=tf.int32),
    }
    scores = saved_model(inputs)

    # Assert model output is a ragged tensor and has the correct contents.
    self.assertIsInstance(scores, tf.RaggedTensor)
    self.assertGreater(scores[0, 0], scores[0, 1])


if __name__ == "__main__":
  tf.test.main()
