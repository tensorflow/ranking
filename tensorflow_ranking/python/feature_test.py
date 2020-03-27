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

"""Tests for feature transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.feature_column import feature_column_lib as feature_column

from tensorflow_ranking.python import feature as feature_lib

EXAMPLE_1_PROTO = """features: {
  feature: { key: "example_0.age" value: { int64_list: { value: [ 41 ] } } }
  feature: { key: "example_0.weight" value: { int64_list: { value: [ 77 ] } } }
  feature: { key: "example_0.label" value: { float_list: { value: [ 1.0 ] } } }
  feature: { key: "example_1.age" value: { int64_list: { value: [ 43 ] } } }
  feature: { key: "example_1.weight" value: { int64_list: { value: [ 78 ] } } }
  feature: { key: "example_1.label" value: { float_list: { value: [ 0.0 ] } } }
}"""

EXAMPLE_2_PROTO = """features: {
  feature: { key: "example_0.age" value: { int64_list: { value: [ 25 ] } } }
  feature: { key: "example_0.weight" value: { int64_list: { value: [ 95 ] } } }
  feature: { key: "example_0.label" value: { float_list: { value: [ 1.0 ] } } }
  feature: { key: "example_1.age" value: { int64_list: { value: [ 53 ] } } }
  feature: { key: "example_1.weight" value: { int64_list: { value: [ 85 ] } } }
  feature: { key: "example_1.label" value: { float_list: { value: [ 0.0 ] } } }
}
"""


def _create_input_fn():

  def my_input_fn():
    feature_to_type = {
        "example_0.age": tf.io.FixedLenFeature([1], tf.int64),
        "example_1.age": tf.io.FixedLenFeature([1], tf.int64),
        "example_0.weight": tf.io.FixedLenFeature([1], tf.int64),
        "example_1.weight": tf.io.FixedLenFeature([1], tf.int64),
        "example_0.label": tf.io.FixedLenFeature([1], tf.float32),
        "example_1.label": tf.io.FixedLenFeature([1], tf.float32)
    }
    feature_1_proto = tf.train.Example()
    feature_2_proto = tf.train.Example()
    text_format.Merge(EXAMPLE_1_PROTO, feature_1_proto)
    text_format.Merge(EXAMPLE_2_PROTO, feature_2_proto)

    features_tensor = tf.io.parse_example(
        serialized=[
            feature_1_proto.SerializeToString(),
            feature_2_proto.SerializeToString()
        ],
        features=feature_to_type)

    # Create the dataset.
    dataset = tf.data.Dataset.from_tensor_slices(features_tensor).batch(2)

    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return my_input_fn


class FeatureLibTest(tf.test.TestCase, parameterized.TestCase):

  def test_make_identity_transform_fn(self):
    with tf.Graph().as_default():
      features = {
          "context":  # Input size: (batch_size=2, num_features=2).
              tf.convert_to_tensor(value=[[1.0, 1.0], [1.0, 1.0]]),
          "per_example":
              tf.convert_to_tensor(value=[[[10.0]], [[10.0]]]),
      }
      with tf.compat.v1.Session() as sess:
        transform_fn = feature_lib.make_identity_transform_fn(["context"])
        context_features, per_example_features = sess.run(
            transform_fn(features, 1))
        self.assertCountEqual(["context"], context_features)
        self.assertAllEqual([[1.0, 1.0], [1.0, 1.0]],
                            context_features["context"])

        self.assertCountEqual(["per_example"], per_example_features)
        self.assertAllEqual([[[10.0]], [[10.0]]],
                            per_example_features["per_example"])

  def test_encode_features(self):
    with tf.Graph().as_default():
      # Inputs.
      vocabulary_size = 4
      # -1 values are ignored.
      input_a = np.array([
          [3, -1, -1],  # example 0, ids [3]
          [0, 1, -1],  # example 1, ids [0, 1]
      ])
      input_b = np.array([
          [0, -1, -1],  # example 0, ids [0]
          [-1, -1, -1],  # example 1, ids []
      ])
      input_features = {"aaa": input_a, "bbb": input_b}

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.),  # id 2
          (9., 13.)  # id 3
      )

      # Expected lookup result, using combiner='mean'.
      expected_lookups_a = (
          # example 0:
          (9., 13.),  # ids [3], embedding = [9, 13]
          # example 1:
          (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
      )
      expected_lookups_b = (
          # example 0:
          (1., 2.),  # ids [0], embedding = [1, 2]
          # example 1:
          (0., 0.),  # ids [], embedding = [0, 0]
      )

      # Build columns.
      categorical_column_a = feature_column.categorical_column_with_identity(
          key="aaa", num_buckets=vocabulary_size)
      categorical_column_b = feature_column.categorical_column_with_identity(
          key="bbb", num_buckets=vocabulary_size)
      embed_column_a, embed_column_b = feature_column.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=lambda shape, dtype, partition_info: embedding_values,
          shared_embedding_collection_name="custom_collection_name")

      feature_columns = {"aaa": embed_column_a, "bbb": embed_column_b}

      cols_to_tensors = feature_lib.encode_features(
          input_features,
          feature_columns.values(),
          mode=tf.estimator.ModeKeys.EVAL)

      embedding_lookup_a = cols_to_tensors[feature_columns["aaa"]]
      embedding_lookup_b = cols_to_tensors[feature_columns["bbb"]]

      # Assert expected embedding variable and lookups.
      global_vars = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
      embedding_var = global_vars[0]
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        self.assertAllEqual(embedding_values, embedding_var.eval())
        self.assertAllEqual(expected_lookups_a, embedding_lookup_a.eval())
        self.assertAllEqual(expected_lookups_b, embedding_lookup_b.eval())

  def test_encode_listwise_features(self):
    with tf.Graph().as_default():
      # Batch size = 2, list_size = 2.
      features = {
          "query_length":
              tf.convert_to_tensor(value=[[1], [2]]),
          "utility":
              tf.convert_to_tensor(value=[[[1.0], [0.0]], [[0.0], [1.0]]]),
          "unigrams":
              tf.SparseTensor(
                  indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                  values=["ranking", "regression", "classification", "ordinal"],
                  dense_shape=[2, 2, 1])
      }
      context_feature_columns = {
          "query_length":
              feature_column.numeric_column(
                  "query_length", shape=(1,), default_value=0, dtype=tf.int64)
      }
      example_feature_columns = {
          "utility":
              feature_column.numeric_column(
                  "utility", shape=(1,), default_value=0.0, dtype=tf.float32),
          "unigrams":
              feature_column.embedding_column(
                  feature_column.categorical_column_with_vocabulary_list(
                      "unigrams",
                      vocabulary_list=[
                          "ranking", "regression", "classification", "ordinal"
                      ]),
                  dimension=10)
      }

      with self.assertRaisesRegexp(
          ValueError,
          r"2nd dimension of tensor must be equal to input size: 3, "
          "but found .*"):
        feature_lib.encode_listwise_features(
            features,
            input_size=3,
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns)

      context_features, example_features = feature_lib.encode_listwise_features(
          features,
          input_size=2,
          context_feature_columns=context_feature_columns,
          example_feature_columns=example_feature_columns)
      self.assertAllEqual(["query_length"], sorted(context_features))
      self.assertAllEqual(["unigrams", "utility"], sorted(example_features))
      self.assertAllEqual([2, 2, 10],
                          example_features["unigrams"].get_shape().as_list())
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        context_features, example_features = sess.run(
            [context_features, example_features])
        self.assertAllEqual([[1], [2]], context_features["query_length"])
        self.assertAllEqual([[[1.0], [0.0]], [[0.0], [1.0]]],
                            example_features["utility"])

  def test_encode_listwise_features_infer_input_size(self):
    with tf.Graph().as_default():
      # Batch size = 2, list_size = 2.
      features = {
          "query_length":
              tf.convert_to_tensor(value=[[1], [2]]),
          "utility":
              tf.convert_to_tensor(value=[[[1.0], [0.0]], [[0.0], [1.0]]]),
          "unigrams":
              tf.SparseTensor(
                  indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                  values=["ranking", "regression", "classification", "ordinal"],
                  dense_shape=[2, 2, 1])
      }
      context_feature_columns = {
          "query_length":
              feature_column.numeric_column(
                  "query_length", shape=(1,), default_value=0, dtype=tf.int64)
      }
      example_feature_columns = {
          "utility":
              feature_column.numeric_column(
                  "utility", shape=(1,), default_value=0.0, dtype=tf.float32),
          "unigrams":
              feature_column.embedding_column(
                  feature_column.categorical_column_with_vocabulary_list(
                      "unigrams",
                      vocabulary_list=[
                          "ranking", "regression", "classification", "ordinal"
                      ]),
                  dimension=10)
      }

      context_features, example_features = feature_lib.encode_listwise_features(
          features,
          context_feature_columns=context_feature_columns,
          example_feature_columns=example_feature_columns)
      self.assertAllEqual(["query_length"], sorted(context_features))
      self.assertAllEqual(["unigrams", "utility"], sorted(example_features))
      self.assertAllEqual([2, 2, 10],
                          example_features["unigrams"].get_shape().as_list())
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        context_features, example_features = sess.run(
            [context_features, example_features])
        self.assertAllEqual([[1], [2]], context_features["query_length"])
        self.assertAllEqual([[[1.0], [0.0]], [[0.0], [1.0]]],
                            example_features["utility"])

  def test_encode_listwise_features_renaming(self):
    """Tests for using different names in feature columns vs features."""
    with tf.Graph().as_default():
      # Batch size = 2, list_size = 2.
      features = {
          "query_length":
              tf.convert_to_tensor(value=[[1], [2]]),
          "utility":
              tf.convert_to_tensor(value=[[[1.0], [0.0]], [[0.0], [1.0]]]),
          "unigrams":
              tf.SparseTensor(
                  indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                  values=["ranking", "regression", "classification", "ordinal"],
                  dense_shape=[2, 2, 1])
      }
      context_feature_columns = {
          "query_length":
              feature_column.numeric_column(
                  "query_length", shape=(1,), default_value=0, dtype=tf.int64)
      }
      example_feature_columns = {
          "utility_renamed":
              feature_column.numeric_column(
                  "utility", shape=(1,), default_value=0.0, dtype=tf.float32),
          "unigrams_renamed":
              feature_column.embedding_column(
                  feature_column.categorical_column_with_vocabulary_list(
                      "unigrams",
                      vocabulary_list=[
                          "ranking", "regression", "classification", "ordinal"
                      ]),
                  dimension=10)
      }

      context_features, example_features = feature_lib.encode_listwise_features(
          features,
          input_size=2,
          context_feature_columns=context_feature_columns,
          example_feature_columns=example_feature_columns)
      self.assertAllEqual(["query_length"], sorted(context_features))
      self.assertAllEqual(["unigrams_renamed", "utility_renamed"],
                          sorted(example_features))
      self.assertAllEqual(
          [2, 2, 10],
          example_features["unigrams_renamed"].get_shape().as_list())
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        context_features, example_features = sess.run(
            [context_features, example_features])
        self.assertAllEqual([[1], [2]], context_features["query_length"])
        self.assertAllEqual([[[1.0], [0.0]], [[0.0], [1.0]]],
                            example_features["utility_renamed"])

  def test_encode_pointwise_features(self):
    with tf.Graph().as_default():
      # Batch size = 2, tf.Example input format.
      features = {
          "query_length":
              tf.convert_to_tensor(value=[[1], [1]]
                                  ),  # Repeated context feature.
          "utility":
              tf.convert_to_tensor(value=[[1.0], [0.0]]),
          "unigrams":
              tf.SparseTensor(
                  indices=[[0, 0], [1, 0]],
                  values=["ranking", "regression"],
                  dense_shape=[2, 1])
      }
      context_feature_columns = {
          "query_length":
              tf.feature_column.numeric_column(
                  "query_length", shape=(1,), default_value=0, dtype=tf.int64)
      }
      example_feature_columns = {
          "utility":
              tf.feature_column.numeric_column(
                  "utility", shape=(1,), default_value=0.0, dtype=tf.float32),
          "unigrams":
              tf.feature_column.embedding_column(
                  feature_column.categorical_column_with_vocabulary_list(
                      "unigrams",
                      vocabulary_list=[
                          "ranking", "regression", "classification", "ordinal"
                      ]),
                  dimension=10)
      }

      (context_features,
       example_features) = feature_lib.encode_pointwise_features(
           features,
           context_feature_columns=context_feature_columns,
           example_feature_columns=example_feature_columns)
      self.assertAllEqual(["query_length"], sorted(context_features))
      self.assertAllEqual(["unigrams", "utility"], sorted(example_features))
      # Unigrams dense tensor has shape: [batch_size=2, list_size=1, dim=10].
      self.assertAllEqual([2, 1, 10],
                          example_features["unigrams"].get_shape().as_list())
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        context_features, example_features = sess.run(
            [context_features, example_features])
        self.assertAllEqual([[1], [1]], context_features["query_length"])
        # Utility tensor has shape: [batch_size=2, list_size=1, 1].
        self.assertAllEqual([[[1.0]], [[0.0]]], example_features["utility"])


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
