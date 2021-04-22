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
"""Tests for Keras model."""

from absl.testing import parameterized
import tensorflow as tf

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

ELWC_PROTO = text_format.Parse(
    """
    context {
      features {
        feature {
          key: "context_1"
          value { int64_list { value: [3, 1, 4, 1] } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "feature_1"
          value { float_list { value: 0.0 } }
        }
        feature {
          key: "feature_2"
          value { float_list { value: 0.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "feature_1"
          value { float_list { value: 1.0 } }
        }
        feature {
          key: "feature_2"
          value { float_list { value: 1.0 } }
        }
      }
    }
    examples {
      features {
        feature {
          key: "feature_1"
          value { float_list { value: 2.0 } }
        }
        feature {
          key: "feature_2"
          value { float_list { value: 2.0 } }
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


class InputCreatorTest(tf.test.TestCase):

  def test_feature_spec_input_creator(self):
    context_feature_spec = {"c_ragged": tf.io.RaggedFeature(dtype=tf.int64)}
    example_feature_spec = {
        "fixed_len":
            tf.io.FixedLenFeature(
                shape=(1,), dtype=tf.float32, default_value=0.0),
        "var_len":
            tf.io.VarLenFeature(dtype=tf.string),
        "ragged":
            tf.io.RaggedFeature(dtype=tf.string)
    }
    input_creator = model_lib.FeatureSpecInputCreator(context_feature_spec,
                                                      example_feature_spec)
    context_inputs, example_inputs = input_creator()
    self.assertAllEqual(context_inputs["c_ragged"].shape, [None, None])
    self.assertAllEqual(example_inputs["fixed_len"].shape, [None, None, 1])
    self.assertAllEqual(example_inputs["var_len"].shape, [None, None, 1])
    self.assertAllEqual(example_inputs["ragged"].shape, [None, None, None])

  def test_type_spec_input_creator(self):
    type_spec = {
        "c_ragged":
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.float32),
        "ragged":
            tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.float32),
        "tensor":
            tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32)
    }
    context_feature_names = ["c_ragged"]
    example_feature_names = ["ragged", "tensor"]
    input_creator = model_lib.TypeSpecInputCreator(type_spec,
                                                   context_feature_names,
                                                   example_feature_names)
    context_inputs, example_inputs = input_creator()
    self.assertAllEqual(context_inputs["c_ragged"].shape, [None, None])
    self.assertAllEqual(example_inputs["ragged"].shape, [None, None, None])
    self.assertAllEqual(example_inputs["tensor"].shape, [None, None, 256])


class PreprocessorTest(tf.test.TestCase):

  def test_preprocessor_with_spec(self):

    def context_embedding(x):
      return tf.reduce_mean(
          tf.keras.layers.Embedding(input_dim=10, output_dim=4)(x), axis=-2)

    preprocess_spec = {
        "context_1": context_embedding,
    }
    default_value_spec = {
        "feature_1": -1.,
    }
    preprocessor = model_lib.PreprocessorWithSpec(preprocess_spec,
                                                  default_value_spec)
    context_features, example_features = preprocessor(
        {
            "context_1": tf.ragged.constant([[3, 1, 4, 1]]),
        },
        example_inputs={
            "feature_1": tf.ragged.constant([[1., 2.]]),
        },
        mask=[[True, True, True]],
    )
    self.assertAllClose(example_features["feature_1"].numpy(), [[1., 2., -1.]])
    self.assertAllClose(context_features["context_1"].numpy(),
                        [[0.01018536, -0.01101774, 0.01906492, 0.02206106]])


class DummyMultiTaskScorer(model_lib.UnivariateScorer):

  def _score_flattened(self, context_features, example_features):
    return {
        "task1": tf.convert_to_tensor([[1.], [2.], [3.]]),
        "task2": tf.convert_to_tensor([[2.], [4.], [6.]]),
    }


class ModelBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self._context_feature_spec = {
        "context_1": tf.io.RaggedFeature(dtype=tf.int64)
    }
    self._example_feature_spec = {
        name:
        tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0.0)
        for name in ["feature_1", "feature_2"]
    }

    def context_embedding(x):
      return tf.reduce_mean(
          tf.keras.layers.Embedding(input_dim=10, output_dim=4)(x), axis=-2)

    preprocess_spec = {
        "context_1":
            context_embedding,
        "feature_1":
            tf.keras.layers.experimental.preprocessing.Normalization(axis=-1),
        "feature_2":
            lambda t: tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
    }

    self._input_creator = model_lib.FeatureSpecInputCreator(
        self._context_feature_spec, self._example_feature_spec)
    self._preprocessor = model_lib.PreprocessorWithSpec(preprocess_spec)

  def test_dnn_model_builder_with_spec(self):
    dnn_scorer = model_lib.DNNScorer(hidden_layer_dims=[10, 10], output_units=1)
    dnn_model = model_lib.ModelBuilder(
        input_creator=self._input_creator,
        preprocessor=self._preprocessor,
        scorer=dnn_scorer,
        mask_feature_name="mask",
        name="test_model",
    ).build()
    output = dnn_model({
        "context_1": tf.ragged.constant([[3, 1, 4, 1]]),
        "feature_1": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "feature_2": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "mask": tf.convert_to_tensor([[True, True, True]]),
    })
    self.assertAllEqual(output.shape.as_list(), [1, 3])

  def test_gam_model_builder_with_spec(self):
    gam_scorer = model_lib.GAMScorer(
        example_hidden_layer_dims=[10, 10], context_hidden_layer_dims=[10, 10])
    gam_model = model_lib.ModelBuilder(
        input_creator=self._input_creator,
        preprocessor=self._preprocessor,
        mask_feature_name="mask",
        scorer=gam_scorer,
        name="test_model").build()
    output = gam_model({
        "context_1": tf.convert_to_tensor([[1]]),
        "feature_1": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "feature_2": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "mask": tf.convert_to_tensor([[True, True, True]]),
    })
    self.assertAllEqual(output.shape.as_list(), [1, 3])

  def test_multi_task_scorer(self):
    mt_model = model_lib.ModelBuilder(
        input_creator=self._input_creator,
        preprocessor=self._preprocessor,
        scorer=DummyMultiTaskScorer(),
        mask_feature_name="mask",
        name="test_model",
    ).build()
    output = mt_model({
        "context_1": tf.ragged.constant([[3, 1, 4, 1]]),
        "feature_1": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "feature_2": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "mask": tf.convert_to_tensor([[True, True, True]]),
    })
    self.assertEqual(len(output), 2)
    self.assertAllEqual(output["task1"].numpy(), [[1., 2., 3.]])
    self.assertAllEqual(output["task2"].numpy(), [[2., 4., 6.]])

  def test_model_to_saved_model_dense_inputs(self):
    dnn_scorer = model_lib.DNNScorer(hidden_layer_dims=[10, 10], output_units=1)
    dnn_model = model_lib.ModelBuilder(
        input_creator=self._input_creator,
        preprocessor=self._preprocessor,
        scorer=dnn_scorer,
        mask_feature_name="mask",
        name="test_model",
    ).build()

    @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.string)])
    def _predict(serialized):
      features = data.parse_from_example_list(
          serialized,
          context_feature_spec=self._context_feature_spec,
          example_feature_spec=self._example_feature_spec,
          mask_feature_name="mask")
      scores = dnn_model(inputs=features, training=False)
      return {"predictions": scores}

    dnn_model.infer_from_proto = _predict

    # Export the model to a SavedModel.
    tf.saved_model.save(
        dnn_model,
        export_dir="/tmp/keras_model",
        signatures={"predict": dnn_model.infer_from_proto})

    # Import ranker from SavedModel.
    imported = tf.saved_model.load("/tmp/keras_model")
    imported_model = imported.signatures["predict"]
    imported_output = imported_model(
        tf.convert_to_tensor([ELWC_PROTO.SerializeToString()]))["predictions"]

    output = dnn_model({
        "context_1": tf.ragged.constant([[3, 1, 4, 1]]),
        "feature_1": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "feature_2": tf.convert_to_tensor([[[0.], [1], [2]]]),
        "mask": tf.convert_to_tensor([[True, True, True]]),
    })
    self.assertAllClose(output.numpy(), imported_output.numpy())


if __name__ == "__main__":
  tf.test.main()
