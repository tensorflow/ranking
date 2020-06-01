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

# Lint as: python3
"""Tests for Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from google.protobuf import text_format

from tensorflow_ranking.python import data
from tensorflow_ranking.python.keras import feature
from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import metrics
from tensorflow_ranking.python.keras import model as model_lib
from tensorflow_ranking.python.keras import network as network_lib

from tensorflow_serving.apis import input_pb2


def _context_feature_columns():
  return {
      "query_length":
          tf.feature_column.numeric_column(
              "query_length", shape=(1,), default_value=0, dtype=tf.int64)
  }


def _example_feature_columns():
  return {
      "utility":
          tf.feature_column.numeric_column(
              "utility", shape=(1,), default_value=0.0, dtype=tf.float32),
      "unigrams":
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_vocabulary_list(
                  "unigrams",
                  vocabulary_list=[
                      "ranking", "regression", "classification", "ordinal"
                  ]),
              dimension=10)
  }


def _features():
  return {
      "query_length":
          tf.convert_to_tensor(value=[[1], [2]]),
      "utility":
          tf.convert_to_tensor(value=[[[1.0], [0.0]], [[0.0], [1.0]]]),
      "unigrams":
          tf.SparseTensor(
              indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
              values=["ranking", "regression", "classification", "ordinal"],
              dense_shape=[2, 2, 1]),
      "example_list_size":
          tf.convert_to_tensor(value=[1, 2])
  }


EXAMPLE_LIST_PROTO_1 = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "query_length"
          value { int64_list { value: 1 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: "ranking" } }
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
          value { bytes_list { value: "classification" } }
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
          value { bytes_list { value: "ordinal" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())


class _DummyUnivariateRankingNetwork(network_lib.UnivariateRankingNetwork):
  """Dummy univariate ranking network with constant scoring function."""

  def score(self, context_features=None, example_features=None, training=True):
    large_batch_size = tf.shape(example_features["utility"])[0]
    return tf.ones(shape=(large_batch_size, 1))


class FunctionalRankingModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FunctionalRankingModelTest, self).setUp()
    self.context_feature_columns = _context_feature_columns()
    self.example_feature_columns = _example_feature_columns()
    self.features = _features()
    self.optimizer = tf.keras.optimizers.Adagrad()
    self.loss = losses.SoftmaxLoss()
    self.metrics = [metrics.NDCGMetric("ndcg_5", topn=5)]

  @parameterized.named_parameters(("without_padding", None, None),
                                  ("with_padding", "example_list_size", None),
                                  ("with_list_size", "example_list_size", 2))
  def test_create_keras_model(self, size_feature_name, list_size):
    network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns)
    ranker = model_lib.create_keras_model(
        network=network,
        loss=self.loss,
        metrics=self.metrics,
        optimizer=self.optimizer,
        size_feature_name=size_feature_name,
        list_size=list_size)
    self.assertEqual(ranker.optimizer, self.optimizer)
    self.assertEqual(ranker.loss, self.loss)
    if size_feature_name:
      self.assertIn("example_list_size", ranker.input_names)
    else:
      self.assertNotIn("example_list_size", ranker.input_names)

  def test_model_to_json(self):
    network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=self.example_feature_columns,
        name="dummy_univariate_ranking_network")
    ranker = model_lib.create_keras_model(
        network=network,
        loss=self.loss,
        metrics=self.metrics,
        optimizer=self.optimizer,
        size_feature_name="example_list_size")

    json_config = ranker.to_json()
    custom_objects = {
        "GenerateMask": feature.GenerateMask,
        "_DummyUnivariateRankingNetwork": _DummyUnivariateRankingNetwork
    }
    restored_ranker = tf.keras.models.model_from_json(
        json_config, custom_objects=custom_objects)
    self.assertAllEqual(restored_ranker(self.features), ranker(self.features))

  def test_model_to_saved_model_dense_inputs(self):
    # TODO: Add SavedModel support for sparse inputs.
    # After adding @tf.function decorator, _predict function breaks for
    # sparse inputs to Keras model.
    example_feature_columns = {}
    example_feature_columns.update(self.example_feature_columns)
    # Remove sparse feature based EmbeddingColumn.
    del example_feature_columns["unigrams"]
    network = _DummyUnivariateRankingNetwork(
        context_feature_columns=self.context_feature_columns,
        example_feature_columns=example_feature_columns)
    ranker = model_lib.create_keras_model(
        network=network,
        loss=self.loss,
        metrics=self.metrics,
        optimizer=self.optimizer,
        size_feature_name="example_list_size")

    context_feature_spec = tf.feature_column.make_parse_example_spec(
        network.context_feature_columns.values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        network.example_feature_columns.values())

    eval_batch_size = 2

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(eval_batch_size,), dtype=tf.string)
    ])
    def _predict(serialized):
      features = data.parse_from_example_list(
          serialized,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          size_feature_name="example_list_size")
      scores = ranker(inputs=features, training=False)
      return {"predictions": scores}

    ranker.infer_from_proto = _predict

    # Export the model to a SavedModel.
    tf.saved_model.save(
        ranker,
        export_dir="/tmp/functional_keras_model",
        signatures={"predict": ranker.infer_from_proto})

    # Import ranker from SavedModel.
    imported = tf.saved_model.load("/tmp/functional_keras_model")
    imported_ranker_predictor = imported.signatures["predict"]
    output = imported_ranker_predictor(
        tf.convert_to_tensor([
            EXAMPLE_LIST_PROTO_1.SerializeToString(),
            EXAMPLE_LIST_PROTO_2.SerializeToString(),
        ]))["predictions"]

    features = {}
    features.update(self.features)
    # TODO: Add SavedModel support for sparse inputs.
    # After adding @tf.function decorator, _predict function breaks for
    # sparse inputs to Keras model. Hence ranker is also created and called
    # only on dense features. Removing "unigrams", a sparse feature.
    del features["unigrams"]
    self.assertAllClose(
        ranker(features, training=False).numpy(), output.numpy())


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
