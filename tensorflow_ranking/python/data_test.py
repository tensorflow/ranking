# Copyright 2024 The TensorFlow Ranking Authors.
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

# Feature name for example list sizes and masks.
_SIZE = "example_list_size"
_MASK = "mask"

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

TF_EXAMPLE_PROTO_1 = text_format.Parse(
    """
    features {
      feature {
          key: "query_length"
          value { int64_list { value: 1 } }
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

CONTEXT_RAGGED_FEATURE_SPEC = {"query_length": tf.io.RaggedFeature(tf.int64)}

EXAMPLE_RAGGED_FEATURE_SPEC = {
    "unigrams": tf.io.RaggedFeature(tf.string),
    "utility": tf.io.FixedLenFeature([1], tf.float32, default_value=[-1.])
}

CONTEXT_SPARSE_FEATURE_SPEC = {
    "query_length": tf.io.VarLenFeature(tf.int64)
}

EXAMPLE_SPARSE_FEATURE_SPEC = {
    "unigrams": tf.io.VarLenFeature(tf.string),
    "utility": tf.io.VarLenFeature(tf.float32)
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

    return next(iter(dataset))

  return example_list_input_fn


class ExampleListTest(tf.test.TestCase):

  def test_decode_as_serialized_example_list(self):
    context_tensor, list_tensor, sizes = (
        data_lib._decode_as_serialized_example_list(
            [EXAMPLE_LIST_PROTO_1.SerializeToString()]))
    self.assertAllEqual(context_tensor.get_shape().as_list(), [1, 1])
    self.assertAllEqual(list_tensor.get_shape().as_list(), [1, 2])
    self.assertAllEqual(sizes, [2])

  def test_parse_from_example_list(self):
    serialized_example_lists = [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ]
    features = data_lib.parse_from_example_list(
        serialized_example_lists,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_example_list_padding(self):
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

    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 3, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"],
                        [[[0.], [1.0], [-1.]], [[0.], [-1.], [-1.]]])

  def test_parse_example_list_with_sizes(self):
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
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features[_SIZE], [2, 1])
    self.assertAllEqual(features[_MASK],
                        [[True, True, False], [True, False, False]])

  def test_parse_from_example_list_truncate(self):
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

    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 1, 1])
    self.assertAllEqual(features["unigrams"].indices, [[0, 0, 0], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values, [b"tensorflow", b"gbdt"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.]], [[0.]]])

  def test_parse_from_example_list_shuffle(self):
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

  def test_parse_example_list_sparse(self):
    serialized_example_lists = [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ]
    features = data_lib.parse_from_example_list(
        serialized_example_lists,
        list_size=3,
        context_feature_spec=CONTEXT_SPARSE_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_SPARSE_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features["query_length"].dense_shape, [2, 1])
    self.assertAllEqual(features["query_length"].indices, [[0, 0], [1, 0]])
    self.assertAllEqual(features["query_length"].values, [3, 2])
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 3, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    self.assertAllEqual(features["utility"].dense_shape, [2, 3, 1])
    self.assertAllEqual(features["utility"].indices,
                        [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    self.assertAllEqual(features["utility"].values, [0., 1., 0.])
    self.assertAllEqual(features[_MASK],
                        [[True, True, False], [True, False, False]])
    self.assertAllEqual(features[_SIZE], [2, 1])


class ExampleListWithRaggedTest(tf.test.TestCase):

  def test_parse_from_example_list(self):
    serialized_example_lists = [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ]
    features = data_lib.parse_from_example_list(
        serialized_example_lists,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(
        features["unigrams"],
        [[[b"tensorflow"], [b"learning", b"to", b"rank"]], [[b"gbdt"], []]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_example_list_padding(self):
    serialized_example_lists = [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ]
    # Padding since list_size 3 is larger than 2.
    features = data_lib.parse_from_example_list(
        serialized_example_lists,
        list_size=3,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(features["unigrams"],
                        [[[b"tensorflow"], [b"learning", b"to", b"rank"], []],
                         [[b"gbdt"], [], []]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"],
                        [[[0.], [1.0], [-1.]], [[0.], [-1.], [-1.]]])

  def test_parse_from_example_list_truncate(self):
    serialized_example_lists = [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ]
    # Truncate number of examples from 2 to 1.
    features = data_lib.parse_from_example_list(
        serialized_example_lists,
        list_size=1,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(features["unigrams"], [[[b"tensorflow"]], [[b"gbdt"]]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.]], [[0.]]])

  def test_parse_from_example_list_shuffle(self):
    serialized_example_lists = [
        EXAMPLE_LIST_PROTO_1.SerializeToString(),
        EXAMPLE_LIST_PROTO_2.SerializeToString()
    ]
    # Trunate number of examples from 2 to 1.
    features = data_lib.parse_from_example_list(
        serialized_example_lists,
        list_size=1,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC,
        shuffle_examples=True,
        seed=1)

    # With `shuffle_examples` and seed=1, the example `tensorflow` and the
    # example `learning to rank` in EXAMPLE_LIST_PROTO_1 switch order. After
    # truncation at list_size=1, only `learning to rank` in
    # EXAMPLE_LIST_PROTO_1 and `gbdt` in EXAMPLE_LIST_PROTO_2 are left in
    # serialized features.
    self.assertAllEqual(features["unigrams"],
                        [[[b"learning", b"to", b"rank"]], [[b"gbdt"]]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[1.]], [[0.]]])

  def test_parse_from_example_list_shape(self):
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
              context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC))
    # Shape can only be checked for non-ragged tensors.
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
    serialized_example_in_example = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ]
    features = data_lib.parse_from_example_in_example(
        serialized_example_in_example,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_example_in_example_shuffle(self):
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
    serialized_example_in_example = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ]
    features = data_lib.parse_from_example_in_example(
        serialized_example_in_example,
        list_size=3,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features[_SIZE], [2, 1])
    self.assertAllEqual(features[_MASK],
                        [[True, True, False], [True, False, False]])

  def test_parse_example_in_example_sparse(self):
    serialized_example_in_example = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ]
    features = data_lib.parse_from_example_in_example(
        serialized_example_in_example,
        list_size=3,
        context_feature_spec=CONTEXT_SPARSE_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_SPARSE_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features["query_length"].dense_shape, [2, 1])
    self.assertAllEqual(features["query_length"].indices, [[0, 0], [1, 0]])
    self.assertAllEqual(features["query_length"].values, [3, 2])
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 3, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    self.assertAllEqual(features["utility"].dense_shape, [2, 3, 1])
    self.assertAllEqual(features["utility"].indices,
                        [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    self.assertAllEqual(features["utility"].values, [0., 1., 0.])
    self.assertAllEqual(features[_MASK],
                        [[True, True, False], [True, False, False]])
    self.assertAllEqual(features[_SIZE], [2, 1])


class ExampleInExampleWithRaggedTest(tf.test.TestCase):

  def test_parse_from_example_in_example(self):
    serialized_example_in_example = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ]
    features = data_lib.parse_from_example_in_example(
        serialized_example_in_example,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(
        features["unigrams"],
        [[[b"tensorflow"], [b"learning", b"to", b"rank"]], [[b"gbdt"], []]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_example_in_example_shuffle(self):
    serialized_example_in_example = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ]
    features = data_lib.parse_from_example_in_example(
        serialized_example_in_example,
        list_size=1,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC,
        shuffle_examples=True,
        seed=1)

    # With `shuffle_examples` and seed=1, the example `tensorflow` and the
    # example `learning to rank` in EXAMPLES_1 switch order. After
    # truncation at list_size=1, only `learning to rank` in EXAMPLES_1
    # and `gbdt` in EXAMPLES_2 are left in serialized features.
    self.assertAllEqual(features["unigrams"],
                        [[[b"learning", b"to", b"rank"]], [[b"gbdt"]]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[1.]], [[0.]]])


class SequenceExampleTest(tf.test.TestCase):

  def test_parse_from_sequence_example(self):
    features = data_lib.parse_from_sequence_example(
        tf.convert_to_tensor(value=[
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
            SEQ_EXAMPLE_PROTO_2.SerializeToString(),
        ]),
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    self.assertCountEqual(features, ["query_length", "unigrams", "utility"])
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.]], [[0.], [-1.]]])
    # Check static shapes for dense tensors.
    self.assertAllEqual([2, 1], features["query_length"].shape)
    self.assertAllEqual([2, 2, 1], features["utility"].shape)

  def test_parse_from_sequence_example_with_sizes(self):
    features = data_lib.parse_from_sequence_example(
        tf.convert_to_tensor(value=[
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
            SEQ_EXAMPLE_PROTO_2.SerializeToString(),
        ]),
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features[_SIZE], [2, 1])
    self.assertAllEqual(features[_MASK], [[True, True], [True, False]])

  def test_parse_from_sequence_example_with_large_list_size(self):
    features = data_lib.parse_from_sequence_example(
        tf.convert_to_tensor(value=[
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        ]),
        list_size=3,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    self.assertCountEqual(features, ["query_length", "unigrams", "utility"])
    self.assertAllEqual(features["query_length"], [[3]])
    self.assertAllEqual(features["unigrams"].dense_shape, [1, 3, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank"])
    self.assertAllEqual(features["utility"], [[[0.], [1.], [-1.]]])
    # Check static shapes for dense tensors.
    self.assertAllEqual([1, 1], features["query_length"].shape)
    self.assertAllEqual([1, 3, 1], features["utility"].shape)

  def test_parse_from_sequence_example_with_small_list_size(self):
    features = data_lib.parse_from_sequence_example(
        tf.convert_to_tensor(value=[
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        ]),
        list_size=1,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    self.assertCountEqual(features, ["query_length", "unigrams", "utility"])
    self.assertAllEqual(features["unigrams"].dense_shape, [1, 1, 3])
    self.assertAllEqual(features["unigrams"].indices, [[0, 0, 0]])
    self.assertAllEqual(features["unigrams"].values, [b"tensorflow"])
    self.assertAllEqual(features["query_length"], [[3]])
    self.assertAllEqual(features["utility"], [[[0.]]])
    # Check static shapes for dense tensors.
    self.assertAllEqual([1, 1], features["query_length"].shape)
    self.assertAllEqual([1, 1, 1], features["utility"].shape)

  def test_parse_from_sequence_example_missing_frame_exception(self):
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

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        # Error from ParseSingleExample:
        r"Unexpected number of elements in feature utility"
        # Error from ParseSequenceExampleV2:
        r"|Name: <unknown>, Key: utility, Index: 1.  "
        r"Number of values != expected.  "
        r"values size: 0 but output shape: \[1\]"):
      data_lib.parse_from_sequence_example(
          tf.convert_to_tensor(value=[missing_frame_proto.SerializeToString()]),
          list_size=2,
          context_feature_spec=None,
          example_feature_spec={"utility": EXAMPLE_FEATURE_SPEC["utility"]})

  def test_parse_from_sequence_example_missing_feature_list(self):
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

    self.assertAllEqual([1, 2, 1], features["utility"].shape)
    self.assertAllEqual([1, 1, 1], features_0["utility"].shape)

  def test_parse_from_sequence_example_sparse(self):
    features = data_lib.parse_from_sequence_example(
        tf.convert_to_tensor(value=[
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
            SEQ_EXAMPLE_PROTO_2.SerializeToString(),
        ]),
        list_size=3,
        context_feature_spec=CONTEXT_SPARSE_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_SPARSE_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features["query_length"].dense_shape, [2, 1])
    self.assertAllEqual(features["query_length"].indices, [[0, 0], [1, 0]])
    self.assertAllEqual(features["query_length"].values, [3, 2])
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 3, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    self.assertAllEqual(features["utility"].dense_shape, [2, 3, 1])
    self.assertAllEqual(features["utility"].indices,
                        [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    self.assertAllEqual(features["utility"].values, [0., 1., 0.])
    self.assertAllEqual(features[_MASK],
                        [[True, True, False], [True, False, False]])
    self.assertAllEqual(features[_SIZE], [2, 1])


class SequenceExampleWithRaggedTest(tf.test.TestCase):

  def test_parse_from_sequence_example(self):
    serialized_example_in_example = [
        SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        SEQ_EXAMPLE_PROTO_2.SerializeToString()
    ]
    features = data_lib.parse_from_sequence_example(
        serialized_example_in_example,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(
        features["unigrams"],
        [[[b"tensorflow"], [b"learning", b"to", b"rank"]], [[b"gbdt"], []]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_parse_from_sequence_example_padding(self):
    serialized_example_lists = [
        SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        SEQ_EXAMPLE_PROTO_2.SerializeToString()
    ]
    # Padding since list_size 3 is larger than 2.
    features = data_lib.parse_from_sequence_example(
        serialized_example_lists,
        list_size=3,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(features["unigrams"],
                        [[[b"tensorflow"], [b"learning", b"to", b"rank"], []],
                         [[b"gbdt"], [], []]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"],
                        [[[0.], [1.0], [-1.]], [[0.], [-1.], [-1.]]])

  def test_parse_from_sequence_example_truncate(self):
    serialized_example_lists = [
        SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        SEQ_EXAMPLE_PROTO_2.SerializeToString()
    ]
    # Truncate number of examples from 2 to 1.
    features = data_lib.parse_from_sequence_example(
        serialized_example_lists,
        list_size=1,
        context_feature_spec=CONTEXT_RAGGED_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC)

    self.assertAllEqual(features["unigrams"], [[[b"tensorflow"]], [[b"gbdt"]]])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.]], [[0.]]])


class RankingDatasetTest(tf.test.TestCase, parameterized.TestCase):

  def test_make_parsing_fn_eie(self):
    parsing_fn = data_lib.make_parsing_fn(
        data_lib.EIE,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)
    serialized_example_in_example = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ]
    features = parsing_fn(serialized_example_in_example)

    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  def test_make_parsing_fn_seq(self):
    parsing_fn = data_lib.make_parsing_fn(
        data_lib.SEQ,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)
    sequence_examples = [
        SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        SEQ_EXAMPLE_PROTO_2.SerializeToString(),
    ]
    features = parsing_fn(sequence_examples)

    self.assertCountEqual(features, ["query_length", "unigrams", "utility"])
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.]], [[0.], [-1.]]])

  def test_make_parsing_fn_exception(self):
    with self.assertRaises(ValueError):
      data_lib.make_parsing_fn(
          "non_existing_format",
          context_feature_spec=CONTEXT_FEATURE_SPEC,
          example_feature_spec=EXAMPLE_FEATURE_SPEC)

  @parameterized.named_parameters(("with_sloppy_ordering", True, False),
                                  ("with_deterministic_ordering", False, False),
                                  ("from_file_list", False, True))
  def test_build_ranking_dataset(self, sloppy_ordering, from_file_list):
    # Save EIE protos in a sstable file in a temp folder.
    serialized_example_in_examples = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ] * 5
    data_dir = self.get_temp_dir()
    data_file = os.path.join(data_dir, "test_ranking_data.tfrecord")
    if tf.io.gfile.exists(data_file):
      tf.io.gfile.remove(data_file)

    with tf.io.TFRecordWriter(data_file) as writer:
      for serialized_eie in serialized_example_in_examples:
        writer.write(serialized_eie)

    if from_file_list:
      data_file = [data_file]

    batched_dataset = data_lib.build_ranking_dataset(
        file_pattern=data_file,
        data_format=data_lib.EIE,
        batch_size=2,
        list_size=2,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC,
        reader=tf.data.TFRecordDataset,
        shuffle=False,
        sloppy_ordering=sloppy_ordering,
        from_file_list=from_file_list)
    features = next(iter(batched_dataset))
    self.assertAllEqual([2, 1], features["query_length"].get_shape().as_list())
    self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

    self.assertAllEqual(
        sorted(features.keys()), ["query_length", "unigrams", "utility"])

    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

  @parameterized.named_parameters(
      ("with_reader_num_threads_autotune", tf.data.experimental.AUTOTUNE),
      ("with_fixed_reader_num_threads", 5))
  def test_build_ranking_dataset_reader_num_threads(self, reader_num_threads):
    # Save EIE protos in a sstable file in a temp folder.
    serialized_example_in_examples = [
        _example_in_example(CONTEXT_1, EXAMPLES_1).SerializeToString(),
        _example_in_example(CONTEXT_2, EXAMPLES_2).SerializeToString(),
    ] * 5
    data_dir = self.get_temp_dir()
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
    features = next(iter(batched_dataset))
    self.assertAllEqual([2, 1], features["query_length"].get_shape().as_list())
    self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

    self.assertAllEqual(
        sorted(features.keys()), ["query_length", "unigrams", "utility"])

    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
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
    # Save protos in a sstable file in a temp folder.
    serialized_sequence_examples = [
        SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        SEQ_EXAMPLE_PROTO_2.SerializeToString()
    ] * 100
    data_dir = self.get_temp_dir()
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

    features = next(iter(batched_dataset))
    self.assertAllEqual(
        sorted(features), ["query_length", "unigrams", "utility"])
    # Check static shapes for dense tensors.
    self.assertAllEqual([2, 1], features["query_length"].get_shape().as_list())
    self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 2, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
    # Check values directly for dense tensors.
    self.assertAllEqual(features["query_length"], [[3], [2]])
    self.assertAllEqual(features["utility"], [[[0.], [1.0]], [[0.], [-1.]]])

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

  def test_sequence_example_serving_input_receiver_fn_sparse(self):
    with tf.Graph().as_default():
      serving_input_receiver_fn = (
          data_lib.build_sequence_example_serving_input_receiver_fn(
              input_size=2,
              context_feature_spec=CONTEXT_SPARSE_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_SPARSE_FEATURE_SPEC))
      serving_input_receiver = serving_input_receiver_fn()
      self.assertCountEqual(serving_input_receiver.features,
                            ["query_length", "unigrams", "utility"])
      self.assertCountEqual(serving_input_receiver.receiver_tensors.keys(),
                            ["sequence_example"])
      with tf.compat.v1.Session() as sess:
        feature_map = sess.run(
            serving_input_receiver.features,
            feed_dict={
                serving_input_receiver.receiver_tensors["sequence_example"]
                .name: [
                    SEQ_EXAMPLE_PROTO_1.SerializeToString(),
                    SEQ_EXAMPLE_PROTO_2.SerializeToString()
                ]
            })
        self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
        self.assertAllEqual(
            feature_map["unigrams"].indices,
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
        self.assertAllEqual(
            feature_map["unigrams"].values,
            [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
        self.assertAllEqual(feature_map["query_length"].dense_shape, [2, 1])
        self.assertAllEqual(feature_map["query_length"].indices,
                            [[0, 0], [1, 0]])
        self.assertAllEqual(feature_map["query_length"].values, [3, 2])
        self.assertAllEqual(feature_map["utility"].dense_shape, [2, 2, 1])
        self.assertAllEqual(feature_map["utility"].indices,
                            [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        self.assertAllEqual(feature_map["utility"].values, [0., 1., 0.])


class TFExampleDatasetTest(tf.test.TestCase):

  def test_parse_from_tf_example(self):
    serialized_examples = [
        TF_EXAMPLE_PROTO_1.SerializeToString(),
        TF_EXAMPLE_PROTO_2.SerializeToString()
    ]
    features = data_lib.parse_from_tf_example(
        serialized=serialized_examples,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features[_SIZE], [1, 1])
    self.assertAllEqual(features[_MASK], [[True], [True]])
    # Test dense_shape, indices and values for a SparseTensor.
    self.assertAllEqual(features["unigrams"].dense_shape, [2, 1, 3])
    self.assertAllEqual(features["unigrams"].indices,
                        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 2]])
    self.assertAllEqual(features["unigrams"].values,
                        [b"tensorflow", b"learning", b"to", b"rank"])
    # For Tensors with dense values, values can be directly checked.
    self.assertAllEqual(features["query_length"], [[1], [3]])
    self.assertAllEqual(features["utility"], [[[0.]], [[1.]]])

  def test_parse_from_tf_example_ragged(self):
    serialized_examples = [
        TF_EXAMPLE_PROTO_1.SerializeToString(),
        TF_EXAMPLE_PROTO_2.SerializeToString()
    ]
    features = data_lib.parse_from_tf_example(
        serialized=serialized_examples,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC,
        size_feature_name=_SIZE,
        mask_feature_name=_MASK)

    self.assertAllEqual(features[_SIZE], [1, 1])
    self.assertAllEqual(features[_MASK], [[True], [True]])
    self.assertAllEqual(features["unigrams"],
                        [[[b"tensorflow"]], [[b"learning", b"to", b"rank"]]])
    self.assertAllEqual(features["query_length"], [[1], [3]])
    self.assertAllEqual(features["utility"], [[[0.]], [[1.]]])

  def test_build_tf_example_serving_input_receiver_fn(self):
    with tf.Graph().as_default():
      serving_input_receiver_fn = (
          data_lib.build_tf_example_serving_input_receiver_fn(
              context_feature_spec=CONTEXT_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_FEATURE_SPEC,
              size_feature_name=_SIZE,
              mask_feature_name=_MASK))
      serving_input_receiver = serving_input_receiver_fn()
      context_features = {
          name: tensor
          for name, tensor in serving_input_receiver.features.items()
          if name in CONTEXT_FEATURE_SPEC
      }
      example_features = {
          name: tensor
          for name, tensor in serving_input_receiver.features.items()
          if name in EXAMPLE_FEATURE_SPEC
      }
      self.assertAllEqual(sorted(context_features.keys()), ["query_length"])
      self.assertAllEqual(
          sorted(example_features.keys()), ["unigrams", "utility"])
      self.assertCountEqual(["input_ranking_data"],
                            serving_input_receiver.receiver_tensors.keys())
      with tf.compat.v1.Session() as sess:
        context_map, example_map = sess.run(
            [context_features, example_features],
            feed_dict={
                serving_input_receiver.receiver_tensors["input_ranking_data"]
                .name: [
                    TF_EXAMPLE_PROTO_1.SerializeToString(),
                    TF_EXAMPLE_PROTO_2.SerializeToString()
                ]
            })
        # Test dense_shape, indices and values for a SparseTensor.
        self.assertAllEqual(example_map["unigrams"].dense_shape, [2, 1, 3])
        self.assertAllEqual(example_map["unigrams"].indices,
                            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 2]])
        self.assertAllEqual(example_map["unigrams"].values,
                            [b"tensorflow", b"learning", b"to", b"rank"])
        # For Tensors with dense values, values can be directly checked.
        self.assertAllEqual(context_map["query_length"], [[1], [3]])
        self.assertAllEqual(example_map["utility"], [[[0.]], [[1.]]])

  def test_build_tf_example_serving_input_receiver_fn_ragged(self):
    with tf.Graph().as_default():
      serving_input_receiver_fn = (
          data_lib.build_tf_example_serving_input_receiver_fn(
              context_feature_spec=CONTEXT_FEATURE_SPEC,
              example_feature_spec=EXAMPLE_RAGGED_FEATURE_SPEC,
              size_feature_name=_SIZE,
              mask_feature_name=_MASK))
      serving_input_receiver = serving_input_receiver_fn()
      context_features = {
          name: tensor
          for name, tensor in serving_input_receiver.features.items()
          if name in CONTEXT_FEATURE_SPEC
      }
      example_features = {
          name: tensor
          for name, tensor in serving_input_receiver.features.items()
          if name in EXAMPLE_FEATURE_SPEC
      }
      self.assertAllEqual(sorted(context_features.keys()), ["query_length"])
      self.assertAllEqual(
          sorted(example_features.keys()), ["unigrams", "utility"])
      self.assertCountEqual(["input_ranking_data"],
                            serving_input_receiver.receiver_tensors.keys())
      with tf.compat.v1.Session() as sess:
        context_map, example_map = sess.run(
            [context_features, example_features],
            feed_dict={
                serving_input_receiver.receiver_tensors["input_ranking_data"]
                .name: [
                    TF_EXAMPLE_PROTO_1.SerializeToString(),
                    TF_EXAMPLE_PROTO_2.SerializeToString()
                ]
            })
        self.assertAllEqual(
            example_map["unigrams"],
            [[[b"tensorflow"]], [[b"learning", b"to", b"rank"]]])
        self.assertAllEqual(context_map["query_length"], [[1], [3]])
        self.assertAllEqual(example_map["utility"], [[[0.]], [[1.]]])


if __name__ == "__main__":
  tf.test.main()
