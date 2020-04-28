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

"""Input data processing tests for ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from tensorflow_ranking.python import data as data_lib
from tensorflow_serving.apis import input_pb2

# Feature name for example list sizes.
_SIZE = "__list_size__"

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

CONTEXT_1 = text_format.Parse(
    """
    features {
      feature {
        key: "query_length"
        value { int64_list { value: 3 } }
      }
    }""", tf.train.Example())

EXAMPLES_1 = [
    text_format.Parse(
        """
    features {
      feature {
        key: "unigrams"
        value { bytes_list { value: "tensorflow" } }
      }
      feature {
        key: "utility"
        value { float_list { value: 0.0 } }
      }
    }""", tf.train.Example()),
    text_format.Parse(
        """
    features {
      feature {
        key: "unigrams"
        value { bytes_list { value: ["learning", "to", "rank"] } }
      }
      feature {
        key: "utility"
        value { float_list { value: 1.0 } }
      }
    }""", tf.train.Example()),
]

CONTEXT_2 = text_format.Parse(
    """
    features {
      feature {
        key: "query_length"
        value { int64_list { value: 2 } }
      }
    }""", tf.train.Example())

EXAMPLES_2 = [
    text_format.Parse(
        """
    features {
      feature {
        key: "unigrams"
        value { bytes_list { value: "gbdt" } }
      }
      feature {
        key: "utility"
        value { float_list { value: 0.0 } }
      }
    }""", tf.train.Example()),
]

SEQ_EXAMPLE_PROTO_1 = text_format.Parse(
    """
    context {
      feature {
        key: "query_length"
        value { int64_list { value: 3 } }
      }
    }
    feature_lists {
      feature_list {
        key: "unigrams"
        value {
          feature { bytes_list { value: "tensorflow" } }
          feature { bytes_list { value: ["learning", "to", "rank"] } }
        }
      }
      feature_list {
        key: "utility"
        value {
          feature { float_list { value: 0.0 } }
          feature { float_list { value: 1.0 } }
        }
      }
    }
    """, tf.train.SequenceExample())

SEQ_EXAMPLE_PROTO_2 = text_format.Parse(
    """
    context {
      feature {
        key: "query_length"
        value { int64_list { value: 2 } }
      }
    }
    feature_lists {
      feature_list {
        key: "unigrams"
        value {
          feature { bytes_list { value: "gbdt" } }
        }
      }
      feature_list {
        key: "utility"
        value {
          feature { float_list { value: 0.0 } }
        }
      }
    }
    """, tf.train.SequenceExample())

CONTEXT_FEATURE_SPEC = {
    "query_length": tf.io.FixedLenFeature([1], tf.int64, default_value=[0])
}

EXAMPLE_FEATURE_SPEC = {
    "unigrams": tf.io.VarLenFeature(tf.string),
    "utility": tf.io.FixedLenFeature([1], tf.float32, default_value=[-1.])
}


def make_example_list_input_fn():
  """example_list input fn."""

  def _example_list_proto_generator():
    return [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ] * 100

  def example_list_input_fn():
    dataset = tf.data.Dataset.from_generator(_example_list_proto_generator,
                                             (tf.string), (tf.TensorShape([])))
    kwargs = {
        "list_size": 2,
        "context_feature_spec": CONTEXT_FEATURE_SPEC,
        "example_feature_spec": EXAMPLE_FEATURE_SPEC,
    }
    dataset = dataset.map(
        functools.partial(data_lib.parse_single_example_list,
                          **kwargs)).batch(2)

    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return example_list_input_fn


class ExampleListTest(tf.test.TestCase):

  def test_decode_as_serialized_example_list(self):
    with tf.Graph().as_default():
      context_tensor, list_tensor, sizes = (
          data_lib._decode_as_serialized_example_list(
              [EXAMPLE_LIST_PROTO_1.SerializeToString()]))
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        context_, list_, sizes_ = sess.run([context_tensor, list_tensor, sizes])
        self.assertAllEqual(
            tf.convert_to_tensor(value=context_).get_shape().as_list(), [1, 1])
        self.assertAllEqual(
            tf.convert_to_tensor(value=list_).get_shape().as_list(), [1, 2])
        self.assertAllEqual(sizes_, [2])

  def test_parse_from_example_list(self):
    with tf.Graph().as_default():
      serialized_example_lists = [
          EXAMPLE_LIST_PROTO_1.SerializeToString(),
          EXAMPLE_LIST_PROTO_2.SerializeToString()
      ]
      features = data_lib.parse_from_example_list(
          serialized_example_lists,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            features["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            features["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_example_list_padding(self):
    with tf.Graph().as_default():
      serialized_example_lists = [
          EXAMPLE_LIST_PROTO_1.SerializeToString(),
          EXAMPLE_LIST_PROTO_2.SerializeToString()
      ]
      # Padding since list_size 3 is larger than 2.
      features = data_lib.parse_from_example_list(
          serialized_example_lists,
          list_size=3,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 3, 3])
        self.assertAllEqual(
            features["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            features["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"],
                            [[[0.], [1.0], [-1.]], [[0.], [-1.], [-1.]]])

  def test_parse_example_list_with_sizes(self):
    with tf.Graph().as_default():
      serialized_example_lists = [
          EXAMPLE_LIST_PROTO_1.SerializeToString(),
          EXAMPLE_LIST_PROTO_2.SerializeToString()
      ]
      # Padding since list_size 3 is larger than 2.
      features = data_lib.parse_from_example_list(
          serialized_example_lists,
          list_size=3,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          size_feature_name=_SIZE)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        self.assertAllEqual(features[_SIZE], [2, 1])

  def test_parse_from_example_list_truncate(self):
    with tf.Graph().as_default():
      serialized_example_lists = [
          EXAMPLE_LIST_PROTO_1.SerializeToString(),
          EXAMPLE_LIST_PROTO_2.SerializeToString()
      ]
      # Truncate number of examples from 2 to 1.
      features = data_lib.parse_from_example_list(
          serialized_example_lists,
          list_size=1,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 1, 1])
        self.assertAllEqual(features["unigrams"].indices,
                            [[0, 0, 0], [1, 0, 0]])
        self.assertAllEqual(features["unigrams"].values,
                            [b"tensorflow", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[0.]], [[0.]]])

  def test_parse_from_example_list_shuffle(self):
    with tf.Graph().as_default():
      serialized_example_lists = [
          EXAMPLE_LIST_PROTO_1.SerializeToString(),
          EXAMPLE_LIST_PROTO_2.SerializeToString()
      ]
      # Trunate number of examples from 2 to 1.
      features = data_lib.parse_from_example_list(
          serialized_example_lists,
          list_size=1,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          shuffle_examples=True,
          seed=1)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # With `shuffle_examples` and seed=1, the example `tensorflow` and the
        # example `learning to rank` in EXAMPLE_LIST_PROTO_1 switch order. After
        # truncation at list_size=1, only `learning to rank` in
        # EXAMPLE_LIST_PROTO_1 and `gbdt` in EXAMPLE_LIST_PROTO_2 are left in
        # serialized features.
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 1, 3])
        self.assertAllEqual(features["unigrams"].indices,
                            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0]])
        self.assertAllEqual(features["unigrams"].values,
                            [b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[1.]], [[0.]]])

  def test_parse_from_example_list_static_shape(self):
    with tf.Graph().as_default():
      serialized_example_lists = [
          EXAMPLE_LIST_PROTO_1.SerializeToString(),
          EXAMPLE_LIST_PROTO_2.SerializeToString()
      ]
      feature_map_list = []
      for list_size in [None, 100, 1]:
        feature_map_list.append(
            data_lib.parse_from_example_list(
                serialized_example_lists,
                list_size=list_size,
                context_feature_spec=CONTEXT_FEATURE_SPEC,
                example_feature_spec=EXAMPLE_FEATURE_SPEC))
      for features in feature_map_list:
        self.assertAllEqual([2, 1],
                            features["query_length"].get_shape().as_list())
      for features, static_shape in zip(feature_map_list, [
          [2, 2, 1],
          [2, 100, 1],
          [2, 1, 1],
      ]):
        self.assertAllEqual(static_shape,
                            features["utility"].get_shape().as_list())


def _example_in_example(context, examples):
  """Returns an Example in Example."""
  example_in_example = tf.train.Example()
  example_in_example.features.feature[
      "serialized_context"].bytes_list.value.append(context.SerializeToString())
  for ex in examples:
    example_in_example.features.feature[
        "serialized_examples"].bytes_list.value.append(ex.SerializeToString())
  return example_in_example


class ExampleInExampleTest(tf.test.TestCase):

  def test_parse_from_example_in_example(self):
    with tf.Graph().as_default():
      serialized_example_in_example = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
          _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
      ]
      features = data_lib.parse_from_example_in_example(
          serialized_example_in_example,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            features["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            features["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_example_in_example_shuffle(self):
    with tf.Graph().as_default():
      serialized_example_in_example = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
          _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
      ]
      features = data_lib.parse_from_example_in_example(
          serialized_example_in_example,
          list_size=1,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          shuffle_examples=True,
          seed=1)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # With `shuffle_examples` and seed=1, the example `tensorflow` and the
        # example `learning to rank` in EXAMPLES_1 switch order. After
        # truncation at list_size=1, only `learning to rank` in EXAMPLES_1
        # and `gbdt` in EXAMPLES_2 are left in serialized features.
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 1, 3])
        self.assertAllEqual(features["unigrams"].indices,
                            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0]])
        self.assertAllEqual(features["unigrams"].values,
                            [b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[1.]], [[0.]]])

  def test_parse_example_in_example_with_sizes(self):
    with tf.Graph().as_default():
      serialized_example_in_example = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
          _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
      ]
      features = data_lib.parse_from_example_in_example(
          serialized_example_in_example,
          list_size=3,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          size_feature_name=_SIZE)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        self.assertAllEqual(features[_SIZE], [2, 1])


class SequenceExampleTest(tf.test.TestCase):

  def test_parse_from_sequence_example(self):
    with tf.Graph().as_default():
      features = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[
              SEQ_EXAMPLE_PROTO_1.SerializeToString(),
              SEQ_EXAMPLE_PROTO_2.SerializeToString(),
          ]),
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(features)
        self.assertEqual(
            sorted(feature_map), ["query_length", "unigrams", "utility"])
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            feature_map["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            feature_map["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        self.assertAllEqual(feature_map["query_length"], [[3], [2]])
        self.assertAllEqual(feature_map["utility"],
                            [[[0.], [1.]], [[0.], [-1.]]])
        # Check static shapes for dense tensors.
        self.assertAllEqual([2, 1], feature_map["query_length"].shape)
        self.assertAllEqual([2, 2, 1], feature_map["utility"].shape)

  def test_parse_from_sequence_example_with_sizes(self):
    with tf.Graph().as_default():
      features = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[
              SEQ_EXAMPLE_PROTO_1.SerializeToString(),
              SEQ_EXAMPLE_PROTO_2.SerializeToString(),
          ]),
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          size_feature_name=_SIZE)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        self.assertAllEqual(features[_SIZE], [2, 1])

  def test_parse_from_sequence_example_with_large_list_size(self):
    with tf.Graph().as_default():
      features = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[
              SEQ_EXAMPLE_PROTO_1.SerializeToString(),
          ]),
          list_size=3,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(features)
        self.assertEqual(
            sorted(feature_map), ["query_length", "unigrams", "utility"])
        self.assertAllEqual(feature_map["query_length"], [[3]])
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [1, 3, 3])
        self.assertAllEqual(feature_map["unigrams"].indices,
                            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2]])
        self.assertAllEqual(feature_map["unigrams"].values,
                            [b"tensorflow", b"learning", b"to", b"rank"])
        self.assertAllEqual(feature_map["utility"], [[[0.], [1.], [-1.]]])
        # Check static shapes for dense tensors.
        self.assertAllEqual([1, 1], feature_map["query_length"].shape)
        self.assertAllEqual([1, 3, 1], feature_map["utility"].shape)

  def test_parse_from_sequence_example_with_small_list_size(self):
    with tf.Graph().as_default():
      features = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[
              SEQ_EXAMPLE_PROTO_1.SerializeToString(),
          ]),
          list_size=1,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(features)
        self.assertEqual(
            sorted(feature_map), ["query_length", "unigrams", "utility"])
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [1, 1, 3])
        self.assertAllEqual(feature_map["unigrams"].indices, [[0, 0, 0]])
        self.assertAllEqual(feature_map["unigrams"].values, [b"tensorflow"])
        self.assertAllEqual(feature_map["query_length"], [[3]])
        self.assertAllEqual(feature_map["utility"], [[[0.]]])
        # Check static shapes for dense tensors.
        self.assertAllEqual([1, 1], feature_map["query_length"].shape)
        self.assertAllEqual([1, 1, 1], feature_map["utility"].shape)

  def test_parse_from_sequence_example_missing_frame_exception(self):
    with tf.Graph().as_default():
      missing_frame_proto = text_format.Parse(
          """
          feature_lists {
            feature_list {
              key: "utility"
              value {
                feature { float_list { value: 0.0 } }
                feature { }
              }
            }
          }
          """, tf.train.SequenceExample())
      features = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[missing_frame_proto.SerializeToString()]),
          list_size=2,
          context_feature_spec=None,
          example_feature_spec={"utility": EXAMPLE_FEATURE_SPEC["utility"]})

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        with self.assertRaisesRegexp(
            tf.errors.InvalidArgumentError,
            # Error from ParseSingleExample:
            r"Unexpected number of elements in feature utility"
            # Error from ParseSequenceExampleV2:
            r"|Name: <unknown>, Key: utility, Index: 1.  "
            r"Number of values != expected.  "
            r"values size: 0 but output shape: \[1\]"):
          sess.run(features)

  def test_parse_from_sequence_example_missing_feature_list(self):
    with tf.Graph().as_default():
      empty_proto = text_format.Parse(
          """
          feature_lists {
            feature_list {
              key: "utility2"
              value {
                feature { float_list { value: 0.0 } }
              }
            }
          }
          """, tf.train.SequenceExample())
      features = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[empty_proto.SerializeToString()]),
          list_size=2,
          context_feature_spec=None,
          example_feature_spec={"utility": EXAMPLE_FEATURE_SPEC["utility"]})

      features_0 = data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[empty_proto.SerializeToString()]),
          context_feature_spec=None,
          example_feature_spec={
              "utility": EXAMPLE_FEATURE_SPEC["utility"],
              "utility2": EXAMPLE_FEATURE_SPEC["utility"]
          })

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map, feature_0_map = sess.run([features, features_0])
        self.assertAllEqual([1, 2, 1], feature_map["utility"].shape)
        self.assertAllEqual([1, 1, 1], feature_0_map["utility"].shape)


class RankingDatasetTest(tf.test.TestCase, parameterized.TestCase):

  def test_make_parsing_fn_eie(self):
    with tf.Graph().as_default():
      parsing_fn = data_lib.make_parsing_fn(
          data_lib.EIE,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)
      serialized_example_in_example = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
          _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
      ]
      features = parsing_fn(serialized_example_in_example)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            features["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            features["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_make_parsing_fn_seq(self):
    with tf.Graph().as_default():
      parsing_fn = data_lib.make_parsing_fn(
          data_lib.SEQ,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)
      sequence_examples = [
          SEQ_EXAMPLE_PROTO_1.SerializeToString(),
          SEQ_EXAMPLE_PROTO_2.SerializeToString(),
      ]
      features = parsing_fn(sequence_examples)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(features)
        self.assertCountEqual(feature_map,
                              ["query_length", "unigrams", "utility"])
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            feature_map["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            feature_map["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        self.assertAllEqual(feature_map["query_length"], [[3], [2]])
        self.assertAllEqual(feature_map["utility"],
                            [[[0.], [1.]], [[0.], [-1.]]])

  def test_make_parsing_fn_exception(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError):
        data_lib.make_parsing_fn(
            "non_existing_format",
            context_feature_spec=CONTEXT_FEATURE_SPEC,
            example_feature_spec=EXAMPLE_FEATURE_SPEC)

  @parameterized.named_parameters(("with_sloppy_ordering", True),
                                  ("with_deterministic_ordering", False))
  def test_build_ranking_dataset(self, sloppy_ordering):
    with tf.Graph().as_default():
      # Save EIE protos in a sstable file in a temp folder.
      serialized_example_in_examples = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
          _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
      ] * 5
      data_dir = tf.compat.v1.test.get_temp_dir()
      data_file = os.path.join(data_dir, "test_ranking_data.tfrecord")
      if tf.io.gfile.exists(data_file):
        tf.io.gfile.remove(data_file)

      with tf.io.TFRecordWriter(data_file) as writer:
        for serialized_eie in serialized_example_in_examples:
          writer.write(serialized_eie)

      batched_dataset = data_lib.build_ranking_dataset(
          file_pattern=data_file,
          data_format=data_lib.EIE,
          batch_size=2,
          list_size=2,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          reader=tf.data.TFRecordDataset,
          shuffle=False,
          sloppy_ordering=sloppy_ordering)
      features = tf.compat.v1.data.make_one_shot_iterator(
          batched_dataset).get_next()
      self.assertAllEqual([2, 1],
                          features["query_length"].get_shape().as_list())
      self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

      self.assertAllEqual(
          sorted(features.keys()), ["query_length", "unigrams", "utility"])

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            features["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            features["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  @parameterized.named_parameters(
      ("with_reader_num_threads_autotune", tf.data.experimental.AUTOTUNE),
      ("with_fixed_reader_num_threads", 5))
  def test_build_ranking_dataset_reader_num_threads(self, reader_num_threads):
    with tf.Graph().as_default():
      # Save EIE protos in a sstable file in a temp folder.
      serialized_example_in_examples = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
          _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
      ] * 5
      data_dir = tf.compat.v1.test.get_temp_dir()
      data_file = os.path.join(data_dir, "test_ranking_data.tfrecord")
      if tf.io.gfile.exists(data_file):
        tf.io.gfile.remove(data_file)

      with tf.io.TFRecordWriter(data_file) as writer:
        for serialized_eie in serialized_example_in_examples:
          writer.write(serialized_eie)

      batched_dataset = data_lib.build_ranking_dataset(
          file_pattern=data_file,
          data_format=data_lib.EIE,
          batch_size=2,
          list_size=2,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          reader=tf.data.TFRecordDataset,
          shuffle=False,
          reader_num_threads=reader_num_threads)
      features = tf.compat.v1.data.make_one_shot_iterator(
          batched_dataset).get_next()
      self.assertAllEqual([2, 1],
                          features["query_length"].get_shape().as_list())
      self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

      self.assertAllEqual(
          sorted(features.keys()), ["query_length", "unigrams", "utility"])

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(features)
        self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            features["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            features["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3], [2]])
        self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_build_ranking_serving_input_receiver_fn(self):
    with tf.Graph().as_default():
      serving_input_receiver_fn = (
          data_lib.build_ranking_serving_input_receiver_fn(
              data_format=data_lib.EIE,
              context_feature_spec=CONTEXT_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_FEATURE_SPEC))
      serving_input_receiver = serving_input_receiver_fn()
      self.assertCountEqual(serving_input_receiver.features.keys(),
                            ["query_length", "unigrams", "utility"])
      self.assertCountEqual(serving_input_receiver.receiver_tensors.keys(),
                            ["input_ranking_data"])
      eie_input = [
          _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString()
      ]
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        features = sess.run(
            serving_input_receiver.features,
            feed_dict={
                serving_input_receiver.receiver_tensors["input_ranking_data"]
                .name:
                    eie_input
            })
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(features["unigrams"].dense_shape, [1, 2, 3])
        self.assertAllEqual(features["unigrams"].indices,
                            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2]])
        self.assertAllEqual(features["unigrams"].values,
                            [b"tensorflow", b"learning", b"to", b"rank"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(features["query_length"], [[3]])
        self.assertAllEqual(features["utility"], [[[0.], [1.]]])

  def test_sequence_example_serving_input_receiver_fn(self):
    with tf.Graph().as_default():
      serving_input_receiver_fn = (
          data_lib.build_sequence_example_serving_input_receiver_fn(
              input_size=2,
              context_feature_spec=CONTEXT_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_FEATURE_SPEC))
      serving_input_receiver = serving_input_receiver_fn()
      self.assertCountEqual(serving_input_receiver.features,
                            ["query_length", "unigrams", "utility"])
      self.assertCountEqual(serving_input_receiver.receiver_tensors.keys(),
                            ["sequence_example"])
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(
            serving_input_receiver.features,
            feed_dict={
                serving_input_receiver.receiver_tensors["sequence_example"]
                .name: [
                    SEQ_EXAMPLE_PROTO_1.SerializeToString(),
                    SEQ_EXAMPLE_PROTO_2.SerializeToString()
                ]
            })
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            feature_map["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            feature_map["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # Check values directly for dense tensors.
        self.assertAllEqual(feature_map["query_length"], [[3], [2]])
        self.assertAllEqual(feature_map["utility"],
                            [[[0.], [1.0]], [[0.], [-1.]]])


class SequenceExampleDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("with_sloppy_ordering", True),
                                  ("with_deterministic_ordering", False))
  def test_read_batched_sequence_example_dataset(self, sloppy_ordering):
    with tf.Graph().as_default():
      # Save protos in a sstable file in a temp folder.
      serialized_sequence_examples = [
          SEQ_EXAMPLE_PROTO_1.SerializeToString(),
          SEQ_EXAMPLE_PROTO_2.SerializeToString()
      ] * 100
      data_dir = tf.compat.v1.test.get_temp_dir()
      data_file = os.path.join(data_dir, "test_sequence_example.tfrecord")
      if tf.io.gfile.exists(data_file):
        tf.io.gfile.remove(data_file)

      with tf.io.TFRecordWriter(data_file) as writer:
        for s in serialized_sequence_examples:
          writer.write(s)

      batched_dataset = data_lib.read_batched_sequence_example_dataset(
          file_pattern=data_file,
          batch_size=2,
          list_size=2,
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC,
          reader=tf.data.TFRecordDataset,
          shuffle=False,
          sloppy_ordering=sloppy_ordering)

      features = tf.compat.v1.data.make_one_shot_iterator(
          batched_dataset).get_next()
      self.assertAllEqual(
          sorted(features), ["query_length", "unigrams", "utility"])
      # Check static shapes for dense tensors.
      self.assertAllEqual([2, 1],
                          features["query_length"].get_shape().as_list())
      self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(features)
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            feature_map["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            feature_map["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # Check values directly for dense tensors.
        self.assertAllEqual(feature_map["query_length"], [[3], [2]])
        self.assertAllEqual(feature_map["utility"],
                            [[[0.], [1.0]], [[0.], [-1.]]])

  def test_sequence_example_serving_input_receiver_fn(self):
    with tf.Graph().as_default():
      serving_input_receiver_fn = (
          data_lib.build_sequence_example_serving_input_receiver_fn(
              input_size=2,
              context_feature_spec=CONTEXT_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_FEATURE_SPEC))
      serving_input_receiver = serving_input_receiver_fn()
      self.assertCountEqual(serving_input_receiver.features,
                            ["query_length", "unigrams", "utility"])
      self.assertCountEqual(serving_input_receiver.receiver_tensors.keys(),
                            ["sequence_example"])
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        feature_map = sess.run(
            serving_input_receiver.features,
            feed_dict={
                serving_input_receiver.receiver_tensors["sequence_example"]
                .name: [
                    SEQ_EXAMPLE_PROTO_1.SerializeToString(),
                    SEQ_EXAMPLE_PROTO_2.SerializeToString()
                ]
            })
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            feature_map["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            feature_map["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        # Check values directly for dense tensors.
        self.assertAllEqual(feature_map["query_length"], [[3], [2]])
        self.assertAllEqual(feature_map["utility"],
                            [[[0.], [1.0]], [[0.], [-1.]]])


class LibSVMUnitTest(tf.test.TestCase, parameterized.TestCase):

  def test_libsvm_parse_line(self):
    data = "1 qid:10 32:0.14 48:0.97  51:0.45"
    qid, features = data_lib._libsvm_parse_line(data)
    self.assertEqual(qid, 10)
    self.assertDictEqual(features, {
        "32": 0.14,
        "48": 0.97,
        "51": 0.45,
        "label": 1.0
    })


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
