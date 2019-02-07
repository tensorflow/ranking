# Copyright 2019 The TensorFlow Ranking Authors.
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

import os
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import queue_runner
from tensorflow.python.util.protobuf import compare
from tensorflow_ranking.python import data as data_lib


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
    """, example_pb2.SequenceExample())

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
          feature { }
        }
      }
      feature_list {
        key: "utility"
        value {
          feature { float_list { value: 0.0 } }
          feature { float_list { value: 0.0 } }
        }
      }
    }
    """, example_pb2.SequenceExample())


CONTEXT_FEATURE_SPEC = {
    "query_length":
        parsing_ops.FixedLenFeature([1], dtypes.int64, default_value=[0])
}

EXAMPLE_FEATURE_SPEC = {
    "unigrams":
        parsing_ops.VarLenFeature(dtypes.string),
    "utility":
        parsing_ops.FixedLenFeature([1], dtypes.float32, default_value=[0.])
}

LIBSVM_DATA = """2 qid:1 1:0.1 3:0.3 4:-0.4
1 qid:1 1:0.12 4:0.24 5:0.5
0 qid:1 2:0.13
"""


class SequenceExampleTest(compare.ProtoAssertions, test.TestCase,
                          parameterized.TestCase):

  def test_parse_from_sequence_example(self):
    features = data_lib.parse_from_sequence_example(
        ops.convert_to_tensor([
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
            SEQ_EXAMPLE_PROTO_2.SerializeToString(),
        ]),
        list_size=2,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    with session.Session() as sess:
      sess.run(variables.local_variables_initializer())
      queue_runner.start_queue_runners()
      feature_map = sess.run(features)
      self.assertEqual(
          sorted(feature_map), ["query_length", "unigrams", "utility"])
      self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
      self.assertAllEqual(
          feature_map["unigrams"].indices,
          [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
      self.assertAllEqual(feature_map["unigrams"].values,
                          [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
      self.assertAllEqual(feature_map["query_length"], [[3], [2]])
      self.assertAllEqual(feature_map["utility"], [[[0.], [1.]], [[0.], [0.]]])

  def test_parse_from_sequence_example_slice(self):
    features = data_lib.parse_from_sequence_example(
        ops.convert_to_tensor([
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        ]),
        list_size=1,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    with session.Session() as sess:
      sess.run(variables.local_variables_initializer())
      queue_runner.start_queue_runners()
      feature_map = sess.run(features)
      self.assertEqual(
          sorted(feature_map), ["query_length", "unigrams", "utility"])
      self.assertAllEqual(feature_map["unigrams"].dense_shape, [1, 1, 3])
      self.assertAllEqual(feature_map["unigrams"].indices, [[0, 0, 0]])
      self.assertAllEqual(feature_map["unigrams"].values, [b"tensorflow"])
      self.assertAllEqual(feature_map["query_length"], [[3]])
      self.assertAllEqual(feature_map["utility"], [[[0.]]])

  def test_parse_from_sequence_example_pad(self):
    features = data_lib.parse_from_sequence_example(
        ops.convert_to_tensor([
            SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        ]),
        list_size=3,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC)

    with session.Session() as sess:
      sess.run(variables.local_variables_initializer())
      queue_runner.start_queue_runners()
      feature_map = sess.run(features)
      self.assertEqual(
          sorted(feature_map), ["query_length", "unigrams", "utility"])
      self.assertAllEqual(feature_map["query_length"], [[3]])
      self.assertAllEqual(feature_map["unigrams"].dense_shape, [1, 3, 3])
      self.assertAllEqual(feature_map["unigrams"].indices,
                          [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2]])
      self.assertAllEqual(feature_map["unigrams"].values,
                          [b"tensorflow", b"learning", b"to", b"rank"])
      self.assertAllEqual(feature_map["utility"], [[[0.], [1.], [0.]]])

  def test_parse_from_sequence_example_missing_framei_exception(self):
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
        """, example_pb2.SequenceExample())
    features = data_lib.parse_from_sequence_example(
        ops.convert_to_tensor([missing_frame_proto.SerializeToString()]),
        list_size=2,
        context_feature_spec=None,
        example_feature_spec={"utility": EXAMPLE_FEATURE_SPEC["utility"]})

    with session.Session() as sess:
      sess.run(variables.local_variables_initializer())
      queue_runner.start_queue_runners()
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"Unexpected number of elements in feature utility"):
        sess.run(features)

  @parameterized.named_parameters(("with_sloppy_ordering", True),
                                  ("with_deterministic_ordering", False))
  def test_read_batched_sequence_example_dataset(self, sloppy_ordering):
    # Save protos in a sstable file in a temp folder.
    serialized_sequence_examples = [
        SEQ_EXAMPLE_PROTO_1.SerializeToString(),
        SEQ_EXAMPLE_PROTO_2.SerializeToString()
    ] * 100
    data_dir = test.get_temp_dir()
    data_file = os.path.join(data_dir, "test_sequence_example.tfrecord")
    if file_io.file_exists(data_file):
      file_io.delete_file(data_file)

    with tf_record.TFRecordWriter(data_file) as writer:
      for s in serialized_sequence_examples:
        writer.write(s)

    batched_dataset = data_lib.read_batched_sequence_example_dataset(
        file_pattern=data_file,
        batch_size=2,
        list_size=2,
        context_feature_spec=CONTEXT_FEATURE_SPEC,
        example_feature_spec=EXAMPLE_FEATURE_SPEC,
        reader=readers.TFRecordDataset,
        shuffle=False,
        sloppy_ordering=sloppy_ordering)

    features = batched_dataset.make_one_shot_iterator().get_next()
    self.assertAllEqual(
        sorted(features), ["query_length", "unigrams", "utility"])
    # Check static shapes for dense tensors.
    self.assertAllEqual([2, 1], features["query_length"].get_shape().as_list())
    self.assertAllEqual([2, 2, 1], features["utility"].get_shape().as_list())

    with session.Session() as sess:
      sess.run(variables.local_variables_initializer())
      queue_runner.start_queue_runners()
      feature_map = sess.run(features)
      # Test dense_shape, indices and values for a SparseTensor.
      self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
      self.assertAllEqual(
          feature_map["unigrams"].indices,
          [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
      self.assertAllEqual(feature_map["unigrams"].values,
                          [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
      # Check values directly for dense tensors.
      self.assertAllEqual(feature_map["query_length"], [[3], [2]])
      self.assertAllEqual(feature_map["utility"], [[[0.], [1.0]], [[0.], [0.]]])

  def test_sequence_example_serving_input_receiver_fn(self):
    serving_input_receiver_fn = (
        data_lib.build_sequence_example_serving_input_receiver_fn(
            input_size=2,
            context_feature_spec=CONTEXT_FEATURE_SPEC,
            example_feature_spec=EXAMPLE_FEATURE_SPEC))
    serving_input_receiver = serving_input_receiver_fn()
    self.assertAllEqual(
        sorted(serving_input_receiver.features),
        ["query_length", "unigrams", "utility"])
    self.assertEqual(
        sorted(serving_input_receiver.receiver_tensors.keys()),
        ["sequence_example"])
    with session.Session() as sess:
      sess.run(variables.local_variables_initializer())
      queue_runner.start_queue_runners()
      feature_map = sess.run(
          serving_input_receiver.features,
          feed_dict={
              serving_input_receiver.receiver_tensors["sequence_example"].name:
                  [
                      SEQ_EXAMPLE_PROTO_1.SerializeToString(),
                      SEQ_EXAMPLE_PROTO_2.SerializeToString()
                  ]
          })
      # Test dense_shape, indices and values for a SparseTensor.
      self.assertAllEqual(feature_map["unigrams"].dense_shape, [2, 2, 3])
      self.assertAllEqual(
          feature_map["unigrams"].indices,
          [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0]])
      self.assertAllEqual(feature_map["unigrams"].values,
                          [b"tensorflow", b"learning", b"to", b"rank", b"gbdt"])
      # Check values directly for dense tensors.
      self.assertAllEqual(feature_map["query_length"], [[3], [2]])
      self.assertAllEqual(feature_map["utility"], [[[0.], [1.0]], [[0.], [0.]]])


class LibSVMUnitTest(test.TestCase, parameterized.TestCase):

  def test_libsvm_parse_line(self):
    data = "1 qid:10 32:0.14 48:0.97  51:0.45"
    qid, features = data_lib._libsvm_parse_line(data)
    self.assertEqual(qid, 10)
    self.assertDictEqual(
        features,
        {"32": 0.14, "48": 0.97, "51": 0.45, "label": 1.0}
    )

  def test_libsvm_generate(self):
    doc_list = [
        {"1": 0.1, "3": 0.3, "4": -0.4, "label": 2.0},
        {"1": 0.12, "4": 0.24, "5": 0.5, "label": 1.0},
        {"2": 0.13, "label": 0.0},
    ]
    want = {
        "1": np.array([[0.1], [0.], [0.12], [0.]], dtype=np.float32),
        "2": np.array([[0.], [0.13], [0.], [0.]], dtype=np.float32),
        "3": np.array([[0.3], [0.], [0.], [0.]], dtype=np.float32),
        "4": np.array([[-0.4], [0.], [0.24], [0.]], dtype=np.float32),
        "5": np.array([[0.], [0.], [0.5], [0.]], dtype=np.float32),
    }

    np.random.seed(10)
    features, labels = data_lib._libsvm_generate(
        num_features=5, list_size=4, doc_list=doc_list)

    self.assertAllEqual(labels, [2.0, 0.0, 1.0, -1.0])
    self.assertAllEqual(
        sorted(features.keys()),
        sorted(want.keys())
    )
    for k in sorted(want):
      self.assertAllEqual(features.get(k), want.get(k))

  def test_libsvm_generator(self):
    data_dir = test.get_temp_dir()
    data_file = os.path.join(data_dir, "test_libvsvm.txt")
    if file_io.file_exists(data_file):
      file_io.delete_file(data_file)

    with open(data_file, "wt") as writer:
      writer.write(LIBSVM_DATA)

    want = {
        "1": np.array([[0.1], [0.], [0.12], [0.]], dtype=np.float32),
        "2": np.array([[0.], [0.13], [0.], [0.]], dtype=np.float32),
        "3": np.array([[0.3], [0.], [0.], [0.]], dtype=np.float32),
        "4": np.array([[-0.4], [0.], [0.24], [0.]], dtype=np.float32),
        "5": np.array([[0.], [0.], [0.5], [0.]], dtype=np.float32),
    }

    reader = data_lib.libsvm_generator(data_file, 5, 4, seed=10)

    for features, labels in reader():
      self.assertAllEqual(labels, [2.0, 0., 1.0, -1.0])
      self.assertAllEqual(
          sorted(features.keys()),
          sorted(want.keys())
      )
      for k in sorted(want):
        self.assertAllEqual(features.get(k), want.get(k))
      break


if __name__ == "__main__":
  test.main()
