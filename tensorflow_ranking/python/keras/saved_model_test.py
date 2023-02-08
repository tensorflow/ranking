# Copyright 2023 The TensorFlow Ranking Authors.
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

"""Tests for saved_model.py."""
import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.python.keras import saved_model as saved_model_lib
from tensorflow_serving.apis import input_pb2

# Feature name for example list masks.
_MASK = "example_list_mask"

EXAMPLE_LIST_PROTO_1 = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "query_length"
          value { int64_list { value: 3 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: "tensorflow" } }
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
          key: "unigrams"
          value { bytes_list { value: ["learning", "to", "rank"] } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

EXAMPLE_LIST_PROTO_2 = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "query_length"
          value { int64_list { value: 2 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: "gbdt" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 0.0 } }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

TF_EXAMPLE_PROTO_1 = text_format.Parse(
    """
    features {
     feature {
        key: "query_length"
        value { int64_list { value: 3 } }
      }
      feature {
        key: "unigrams"
        value { bytes_list { value: "tensorflow" } }
      }
      feature {
        key: "utility"
        value { float_list { value: 0.0 } }
      }
    }
    """, tf.train.Example())

TF_EXAMPLE_PROTO_2 = text_format.Parse(
    """
    features {
      feature {
        key: "query_length"
        value { int64_list { value: 3 } }
      }
      feature {
        key: "unigrams"
        value { bytes_list { value: ["learning", "to", "rank"] } }
      }
      feature {
        key: "utility"
        value { float_list { value: 1.0 } }
      }
    }
    """, tf.train.Example())

CONTEXT_FEATURE_SPEC = {
    "query_length": tf.io.FixedLenFeature([1], tf.int64, default_value=[0])
}

EXAMPLE_FEATURE_SPEC = {
    "unigrams": tf.io.RaggedFeature(tf.string),
    "utility": tf.io.FixedLenFeature([1], tf.float32, default_value=[-1.])
}


def _create_test_ranking_models():
  inputs = {}
  inputs["query_length"] = tf.keras.Input(
      name="query_length", shape=(1,), dtype=tf.int64)
  inputs["unigrams"] = tf.keras.Input(
      name="unigrams", shape=(None, 1), dtype=tf.string, ragged=True)
  inputs["utility"] = tf.keras.Input(
      name="utility", shape=(None, 1), dtype=tf.float32)
  inputs[_MASK] = tf.keras.Input(name=_MASK, shape=(None,), dtype=tf.bool)
  features = inputs

  # Create dummy logits.
  logits = tf.ones_like(features["utility"])
  logits = tf.squeeze(logits, axis=-1)
  scores = -tf.ones_like(features["utility"])
  scores = tf.squeeze(scores, axis=-1)
  model = tf.keras.Model(
      inputs=inputs, outputs=logits, name="dummy_ranking_model")
  model_with_dict_output = tf.keras.Model(
      inputs=inputs,
      outputs={
          "logits": logits,
          "scores": scores
      },
      name="dummy_ranking_model_with_dict_output")
  model_with_list_output = tf.keras.Model(
      inputs=inputs,
      outputs=[logits, scores],
      name="dummy_ranking_model_with_list_output")
  return model, model_with_dict_output, model_with_list_output


class SignaturesTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_model_path = self.create_tempdir().full_path
    (self._model, self._model_with_dict_output,
     self._model_with_list_output) = _create_test_ranking_models()

  def _export(self, model, filepath, serving_default="regress"):
    signatures = saved_model_lib.Signatures(
        model=model,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC,
        mask_feature_name=_MASK)(
            serving_default)
    model.save(filepath=filepath, signatures=signatures, save_format="tf")

  def test_call_by_saved_model(self):
    self._export(self._model, self._saved_model_path)
    self.assertTrue(tf.saved_model.contains_saved_model(self._saved_model_path))

    saved_model = tf.saved_model.load(export_dir=self._saved_model_path)
    elwc_predictor = saved_model.signatures[tf.saved_model.PREDICT_METHOD_NAME]
    listwise_logits = elwc_predictor(
        tf.convert_to_tensor([
            EXAMPLE_LIST_PROTO_1.SerializeToString(),
            EXAMPLE_LIST_PROTO_1.SerializeToString()
        ]))[tf.saved_model.PREDICT_OUTPUTS]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())

    tf_example_predictor = saved_model.signatures[
        tf.saved_model.REGRESS_METHOD_NAME]
    pointwise_logits = tf_example_predictor(
        tf.convert_to_tensor([
            TF_EXAMPLE_PROTO_1.SerializeToString(),
            TF_EXAMPLE_PROTO_2.SerializeToString()
        ]))[tf.saved_model.REGRESS_OUTPUTS]
    self.assertAllEqual([2], pointwise_logits.get_shape().as_list())
    self.assertAllClose(pointwise_logits, listwise_logits[0])

    del pointwise_logits
    serving_default = saved_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    pointwise_logits = serving_default(
        tf.convert_to_tensor([
            TF_EXAMPLE_PROTO_1.SerializeToString(),
            TF_EXAMPLE_PROTO_2.SerializeToString()
        ]))[tf.saved_model.REGRESS_OUTPUTS]
    self.assertAllEqual([2], pointwise_logits.get_shape().as_list())
    self.assertAllClose(pointwise_logits, listwise_logits[0])

  def test_serving_default_equals_predict(self):
    self._export(self._model, self._saved_model_path, serving_default="predict")
    self.assertTrue(tf.saved_model.contains_saved_model(self._saved_model_path))

    saved_model = tf.saved_model.load(export_dir=self._saved_model_path)
    elwc_predictor = saved_model.signatures[tf.saved_model.PREDICT_METHOD_NAME]
    listwise_logits = elwc_predictor(
        tf.convert_to_tensor([
            EXAMPLE_LIST_PROTO_1.SerializeToString(),
            EXAMPLE_LIST_PROTO_1.SerializeToString()
        ]))[tf.saved_model.PREDICT_OUTPUTS]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())

    tf_example_predictor = saved_model.signatures[
        tf.saved_model.REGRESS_METHOD_NAME]
    pointwise_logits = tf_example_predictor(
        tf.convert_to_tensor([
            TF_EXAMPLE_PROTO_1.SerializeToString(),
            TF_EXAMPLE_PROTO_2.SerializeToString()
        ]))[tf.saved_model.REGRESS_OUTPUTS]
    self.assertAllEqual([2], pointwise_logits.get_shape().as_list())
    self.assertAllClose(pointwise_logits, listwise_logits[0])

    del listwise_logits
    serving_default = saved_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    listwise_logits = serving_default(
        tf.convert_to_tensor([
            EXAMPLE_LIST_PROTO_1.SerializeToString(),
            EXAMPLE_LIST_PROTO_1.SerializeToString()
        ]))[tf.saved_model.PREDICT_OUTPUTS]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())
    self.assertAllClose(pointwise_logits, listwise_logits[0])

  def test_model_can_be_loaded_correctly(self):
    self._export(self._model, self._saved_model_path)
    restored_model = tf.keras.models.load_model(self._saved_model_path)
    self.assertIsInstance(restored_model, tf.keras.Model)
    self.assertDictEqual(self._model.get_config(), restored_model.get_config())

  def test_dict_output_should_be_handled(self):
    self._export(self._model_with_dict_output, self._saved_model_path)
    self.assertTrue(tf.saved_model.contains_saved_model(self._saved_model_path))

    saved_model = tf.saved_model.load(export_dir=self._saved_model_path)
    elwc_predictor = saved_model.signatures[tf.saved_model.PREDICT_METHOD_NAME]
    listwise_outputs = elwc_predictor(
        tf.convert_to_tensor([
            EXAMPLE_LIST_PROTO_1.SerializeToString(),
            EXAMPLE_LIST_PROTO_1.SerializeToString()
        ]))
    listwise_logits = listwise_outputs["logits"]
    listwise_scores = listwise_outputs["scores"]
    self.assertAllEqual([2, 2], listwise_logits.get_shape().as_list())
    self.assertAllEqual([2, 2], listwise_scores.get_shape().as_list())

    tf_example_predictor = saved_model.signatures[
        tf.saved_model.REGRESS_METHOD_NAME]
    pointwise_outputs = tf_example_predictor(
        tf.convert_to_tensor([
            TF_EXAMPLE_PROTO_1.SerializeToString(),
            TF_EXAMPLE_PROTO_2.SerializeToString()
        ]))
    pointwise_logits = pointwise_outputs["logits"]
    pointwise_scores = pointwise_outputs["scores"]
    self.assertAllEqual([2], pointwise_logits.get_shape().as_list())
    self.assertAllClose(pointwise_logits, listwise_logits[0])
    self.assertAllEqual([2], pointwise_scores.get_shape().as_list())
    self.assertAllClose(pointwise_scores, listwise_scores[0])

  def test_model_with_dict_output_can_be_loaded(self):
    self._export(self._model_with_dict_output, self._saved_model_path)
    self.assertTrue(tf.saved_model.contains_saved_model(self._saved_model_path))
    restored_model = tf.keras.models.load_model(self._saved_model_path)
    self.assertIsInstance(restored_model, tf.keras.Model)
    self.assertDictEqual(self._model_with_dict_output.get_config(),
                         restored_model.get_config())

  def test_model_with_list_output_raises_error(self):
    with self.assertRaises(TypeError):
      self._export(self._model_with_list_output, self._saved_model_path)


if __name__ == "__main__":
  tf.test.main()
