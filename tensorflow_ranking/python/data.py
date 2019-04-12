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

"""Input data parsing for ranking library.

Supports data stored in SequenceExample proto format.

SequenceExample (`tf.SequenceExample`) is defined in:
tensorflow/core/example/example.proto
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import six

from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import gfile

from tensorflow.python.estimator.export import export as export_lib

# The document relevance label.
_LABEL_FEATURE = "label"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1.


def parse_from_sequence_example(serialized,
                                list_size,
                                context_feature_spec=None,
                                example_feature_spec=None):
  """Parses SequenceExample to feature maps.

  Args:
    serialized: (Tensor) A string Tensor for a batch of serialized
      SequenceExample.
    list_size: (int) number of required frames in a SequenceExample. This is
      needed to normalize output tensor shapes across batches.
    context_feature_spec: (dict) A mapping from feature keys to
      `FixedLenFeature` or `VarLenFeature` values for context.
    example_feature_spec: (dict) A mapping from feature keys to
      `FixedLenFeature` or `VarLenFeature` values for the list of examples.
      These features are stored in the `feature_lists` field in SequenceExample.
      `FixedLenFeature` is translated to `FixedLenSequenceFeature` to parse
      SequenceExample. Note that no missing value in the middle of a
      `feature_list` is allowed for frames.

  Returns:
    A mapping from feature keys to `Tensor` or `SparseTensor`.
  """
  # Convert `FixedLenFeature` in `example_feature_spec` to
  # `FixedLenSequenceFeature` to parse the `feature_lists` in SequenceExample.
  # TODO(xuanhui): Handle missing feature_list since allow_missing=True.
  fixed_len_sequence_features = {
      k: parsing_ops.FixedLenSequenceFeature(
          s.shape, s.dtype, allow_missing=True)
      for k, s in six.iteritems(example_feature_spec)
      if isinstance(s, parsing_ops.FixedLenFeature)
  }
  sequence_features = example_feature_spec.copy()
  sequence_features.update(fixed_len_sequence_features)
  context, examples, _ = parsing_ops.parse_sequence_example(
      serialized,
      context_features=context_feature_spec,
      sequence_features=sequence_features)

  features = {}
  features.update(context)
  # Slice or pad example features to normalize the tensor shape:
  # [batch_size, num_frames, ...] --> [batch_size, list_size, ...]
  for k, t in six.iteritems(examples):
    # Old shape: [batch_size, num_frames, ...]
    shape = array_ops.unstack(array_ops.shape(t))
    ndims = len(shape)
    num_frames = shape[1]
    # New shape: [batch_size, list_size, ...]
    new_shape = array_ops.concat([[shape[0], list_size], shape[2:]], 0)

    def slice_fn(t=t, ndims=ndims, new_shape=new_shape):
      """Slices the tensor."""
      if isinstance(t, sparse_tensor.SparseTensor):
        return sparse_ops.sparse_slice(t, [0] * ndims,
                                       math_ops.to_int64(new_shape))
      else:
        return array_ops.slice(t, [0] * ndims, new_shape)

    def pad_fn(k=k,
               t=t,
               ndims=ndims,
               num_frames=num_frames,
               new_shape=new_shape):
      """Pads the tensor."""
      if isinstance(t, sparse_tensor.SparseTensor):
        return sparse_ops.sparse_reset_shape(t, new_shape)
      else:
        # Padding is n * 2 tensor where n is the ndims or rank of the padded
        # tensor.
        paddings = array_ops.stack([[0, 0], [0, list_size - num_frames]] +
                                   [[0, 0]] * (ndims - 2))
        return array_ops.pad(
            t,
            paddings,
            constant_values=array_ops.squeeze(
                example_feature_spec[k].default_value[0]))

    tensor = control_flow_ops.cond(num_frames > list_size, slice_fn, pad_fn)
    # Infer static shape for Tensor.
    if not isinstance(tensor, sparse_tensor.SparseTensor):
      static_shape = t.get_shape().as_list()
      static_shape[1] = list_size
      tensor.set_shape(static_shape)
    features[k] = tensor
  return features


def read_batched_sequence_example_dataset(file_pattern,
                                          batch_size,
                                          list_size,
                                          context_feature_spec,
                                          example_feature_spec,
                                          reader=readers.TFRecordDataset,
                                          reader_args=None,
                                          num_epochs=None,
                                          shuffle=True,
                                          shuffle_buffer_size=1000,
                                          shuffle_seed=None,
                                          prefetch_buffer_size=32,
                                          reader_num_threads=10,
                                          sloppy_ordering=True,
                                          drop_final_batch=False):
  """Returns a `Dataset` of features from `SequenceExample`.

  Example:

  ```
  data = [
    sequence_example {
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
            feature { bytes_list { value: ["learning" "to" "rank"] } }
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
    }
    sequence_example {
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
    }
  ]
  ```

  We can use arguments:

  ```
  context_features: {
    "query_length": parsing_ops.FixedenFeature([1], dtypes.int64)
  }
  example_features: {
    "unigrams": parsing_ops.VarLenFeature(dtypes.string),
    "utility": parsing_ops.FixedLenFeature([1], dtypes.float32,
    default_value=[0.])
  }
  batch_size: 2
  ```

  And the expected output is:

  ```python
  {
    "unigrams": SparseTensor(
      indices=array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1,
      1, 0], [1, 1, 1]]),
      values=["tensorflow", "learning", "to", "rank", "gbdt"],
      dense_shape=array([2, 2, 3])),
    "utility": [[[ 0.], [ 1.]], [[ 0.], [ 0.]]],
    "query_length": [[3], [2]],
  }
  ```

  Args:
    file_pattern: (str | list(str)) List of files or patterns of file paths
      containing tf.SequenceExample protos. See `tf.gfile.Glob` for pattern
      rules.
    batch_size: (int) Number of records to combine in a single batch.
    list_size: (int) Number of required examples per SequenceExample. Required
      so that we can normalize tensor shapes across batches.
    context_feature_spec: (dict) A mapping from  feature keys to
      `FixedLenFeature` or `VarLenFeature` values.
    example_feature_spec: (dict) A mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    reader: A function or class that can be called with a `filenames` tensor and
      (optional) `reader_args` and returns a `Dataset`. Defaults to
      `tf.data.TFRecordDataset`.
    reader_args: (list) Additional argument list to pass to the reader class.
    num_epochs: (int) Number of times to read through the dataset. If None,
      cycles through the dataset forever. Defaults to `None`.
    shuffle: (bool) Indicates whether the input should be shuffled. Defaults to
      `True`.
    shuffle_buffer_size: (int) Buffer size of the ShuffleDataset. A large
      capacity ensures better shuffling but would increase memory usage and
      startup time.
    shuffle_seed: (int) Randomization seed to use for shuffling.
    prefetch_buffer_size: (int) Number of feature batches to prefetch in order
      to improve performance. Recommended value is the number of batches
      consumed per training step (default is 1).
    reader_num_threads: (int) Number of threads used to read records. If greater
      than 1, the results will be interleaved.
    sloppy_ordering: (bool) If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order of
      elements after shuffling is deterministic). Defaults to `False`.
    drop_final_batch: (bool) If `True`, and the batch size does not evenly
      divide the input dataset size, the final smaller batch will be dropped.
      Defaults to `True`. If `True`, the batch_size can be statically inferred.

  Returns:
    A dataset of `dict` elements. Each `dict` maps feature keys to
    `Tensor` or `SparseTensor` objects. The context features are mapped to a
    rank-2 tensor of shape [batch_size, feature_size], and the example features
    are mapped to a rank-3 tensor of shape [batch_size, list_size,
    feature_size], where list_size is the number of required example.
  """
  # TODO(xuanhui): Move the file reading part into a common function for all
  # batch readers.
  files = dataset_ops.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  reader_args = reader_args or []
  dataset = files.apply(
      interleave_ops.parallel_interleave(
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          sloppy=sloppy_ordering))

  # Extract values if tensors are stored as key-value tuples. This happens when
  # the reader is tf.data.SSTableDataset.
  if dataset.output_types == (dtypes.string, dtypes.string):
    dataset = dataset.map(lambda _, v: v)

  # Repeat and shuffle, if needed.
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)
  if shuffle:
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size, seed=shuffle_seed)

  # Apply batching. If drop_remainder is True, allows for static inference of
  # batch size.
  dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)

  # Parse batched SequenceExample.
  kwargs = {
      "list_size": list_size,
      "context_feature_spec": context_feature_spec,
      "example_feature_spec": example_feature_spec,
  }
  dataset = dataset.map(
      functools.partial(parse_from_sequence_example, **kwargs))

  # Prefetching allows for data fetching to happen on host while model runs
  # on the accelerator. When run on CPU, makes data fecthing asynchronous.
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

  return dataset


def build_sequence_example_serving_input_receiver_fn(input_size,
                                                     context_feature_spec,
                                                     example_feature_spec,
                                                     default_batch_size=None):
  """Creates a serving_input_receiver_fn for `SequenceExample` inputs.

  A string placeholder is used for inputs. Note that the context_feature_spec
  and example_feature_spec shouldn't contain weights, labels or training
  only features in general.

  Args:
    input_size: (int) number of examples in an tf.SequenceExample. This is
      used for normalize SequenceExample.
    context_feature_spec: (dict) Map from feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    example_feature_spec: (dict) Map from  feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    default_batch_size: (int) Number of query examples expected per batch. Leave
      unset for variable batch size (recommended).

  Returns:
    A `tf.estimator.export.ServingInputReceiver` object, which packages the
    placeholders and the resulting feature Tensors together.
  """

  def serving_input_receiver_fn():
    """An input function on serialized SequenceExample protos."""
    serialized_sequence_example = array_ops.placeholder(
        dtype=dtypes.string,
        shape=[default_batch_size],
        name="input_sequence_example_tensor")
    receiver_tensors = {"sequence_example": serialized_sequence_example}
    features = parse_from_sequence_example(
        serialized_sequence_example,
        input_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec)

    return export_lib.ServingInputReceiver(features, receiver_tensors)

  return serving_input_receiver_fn


def _libsvm_parse_line(libsvm_line):
  """Parses a single LibSVM line to a query ID and a feature dictionary.

  Args:
    libsvm_line: (string) input line in LibSVM format.

  Returns:
    A tuple of query ID and a dict mapping from feature ID (string) to value
    (float). "label" is a special feature ID that represents the relevance
    grade.
  """
  tokens = libsvm_line.split()
  qid = int(tokens[1].split(":")[1])

  features = {_LABEL_FEATURE: float(tokens[0])}
  key_values = [key_value.split(":") for key_value in tokens[2:]]
  features.update({key: float(value) for (key, value) in key_values})

  return qid, features


def _libsvm_generate(num_features, list_size, doc_list):
  """Unpacks a list of document features into `Tensor`s.

  Args:
    num_features: An integer representing the number of features per instance.
    list_size: Size of the document list per query.
    doc_list: A list of dictionaries (one per document) where each
      dictionary is a mapping from feature ID (string) to feature value (float).
  Returns:
    A tuple consisting of a dictionary (feature ID to `Tensor`s) and a label
    `Tensor`.
  """
  # Construct output variables.
  features = {}
  for fid in range(num_features):
    features[str(fid+1)] = np.zeros([list_size, 1], dtype=np.float32)
  labels = np.ones([list_size], dtype=np.float32)*(_PADDING_LABEL)

  # Shuffle the document list and trim to a prescribed list_size.
  np.random.shuffle(doc_list)

  if len(doc_list) > list_size:
    doc_list = doc_list[:list_size]

  # Fill in the output Tensors with feature and label values.
  for idx, doc in enumerate(doc_list):
    for feature_id, value in six.iteritems(doc):
      if feature_id == _LABEL_FEATURE:
        labels[idx] = value
      else:
        features.get(feature_id)[idx, 0] = value

  return features, labels


def libsvm_generator(path, num_features, list_size, seed=None):
  """Parses a LibSVM-formatted input file and aggregates data points by qid.

  Args:
    path: (string) path to dataset in the LibSVM format.
    num_features: An integer representing the number of features per instance.
    list_size: Size of the document list per query.
    seed: Randomization seed used when shuffling the document list.

  Returns:
    A generator function that can be passed to tf.data.Dataset.from_generator().
  """
  if seed is not None:
    np.random.seed(seed)

  def inner_generator():
    """Produces a generator ready for tf.data.Dataset.from_generator.

    It is assumed that data points in a LibSVM-formatted input file are
    sorted by query ID before being presented to this function. This
    assumption simplifies the parsing and aggregation logic: We consume
    lines sequentially and accumulate query-document features until a
    new query ID is observed, at which point the accumulated data points
    are massaged into a tf.data.Dataset compatible representation.

    Yields:
      A tuple of feature and label `Tensor`s.
    """
    # A buffer where observed query-document features will be stored.
    # It is a list of dictionaries, one per query-document pair, where
    # each dictionary is a mapping from a feature ID to a feature value.
    doc_list = []

    with gfile.Open(path, "r") as f:
      # cur indicates the current query ID.
      cur = -1

      for line in f:
        qid, doc = _libsvm_parse_line(line)
        if cur < 0:
          cur = qid

        # If qid is not new store the data and move onto the next line.
        if qid == cur:
          doc_list.append(doc)
          continue

        yield _libsvm_generate(
            num_features, list_size, doc_list)

        # Reset current pointer and re-initialize document list.
        cur = qid
        doc_list = [doc]

    yield _libsvm_generate(num_features, list_size, doc_list)

  return inner_generator
