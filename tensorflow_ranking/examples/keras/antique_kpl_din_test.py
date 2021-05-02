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
import tensorflow_ranking as tfr
from tensorflow_ranking.examples.keras import antique_kpl_din
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

VOCAB = [
    "this", "is", "a", "relevant", "question", "answer", "irrelevant", "data"
]


class ModelTrainAndEvaluateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="use_din", use_document_interaction=True),
      dict(testcase_name="no_use_din", use_document_interaction=False),
  )
  def test_train_and_eval(self, use_document_interaction):
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
        test_input_pattern=data_file,
        model_dir=model_dir,
        vocab_file_path=vocab_file,
        train_batch_size=2,
        eval_batch_size=2,
        num_epochs=1,
        vocab_size=len(VOCAB),
        hidden_layer_dims=[16, 8],
        num_train_steps=2,
        use_document_interaction=use_document_interaction):
      model = antique_kpl_din.train_and_eval()
      context_feature_spec, example_feature_spec = antique_kpl_din._create_feature_spec(
      )
      features = tfr.data.parse_from_example_list(
          [ELWC.SerializeToString(),
           ELWC.SerializeToString()],
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          mask_feature_name=antique_kpl_din._MASK)

      # Check SavedModel is exported correctly.
      saved_model_path = os.path.join(model_dir, "export/")
      self.assertTrue(tf.saved_model.contains_saved_model(saved_model_path))

    # Check SavedModel can be loaded and called correctly.
    saved_model = tf.keras.models.load_model(saved_model_path)
    elwc_predictor = saved_model.signatures[tf.saved_model.PREDICT_METHOD_NAME]
    listwise_logits = elwc_predictor(
        tf.convert_to_tensor(
            [ELWC.SerializeToString(),
             ELWC.SerializeToString()]))[tf.saved_model.PREDICT_OUTPUTS]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())

    # Check SavedModel predictions are deterministic.
    saved_model_pred_1 = saved_model(features, training=False)
    saved_model_pred_2 = saved_model(features, training=False)
    self.assertAllEqual(saved_model_pred_1, saved_model_pred_2)

    # Check model predictions are deterministic.
    keras_model_pred_1 = model(features, training=False)
    keras_model_pred_2 = model(features, training=False)
    self.assertAllEqual(keras_model_pred_1, keras_model_pred_2)

    # Check SavedModel predictions match with keras model.
    keras_model_pred = model(features, training=False)
    saved_model_pred = saved_model(features, training=False)
    self.assertAllEqual(keras_model_pred, saved_model_pred)


if __name__ == "__main__":
  tf.test.main()
