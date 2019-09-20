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

"""Input data parsing for tf-ranking library.

Supports the following data formats:
  - tf.train.SequenceExample
  - tf.train.Example in tf.train.Example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import numpy as np
import six

import tensorflow as tf

from tensorflow_ranking.python import utils

# The document relevance label.
_LABEL_FEATURE = "label"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1.

### RankingDataFormat. ###
# For ExampleInExample.
EIE = "example_in_example"
# For SequenceExample.
SEQ = "sequence_example"


class _RankingDataParser(object):
  """Interface for ranking data parser."""

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               list_size=None,
               context_feature_spec=None,
               example_feature_spec=None):
    """Constructor."""
    if not example_feature_spec:
      raise ValueError("example_feature_spec {} must not be empty.".format(
          example_feature_spec))
    if list_size is None or list_size <= 0:
      self._list_size = None
    else:
      self._list_size = list_size
    self._context_feature_spec = context_feature_spec
    self._example_feature_spec = example_feature_spec

  @abc.abstractmethod
  def parse(self, serialized):
    """Parses a serialzed proto into a feature map."""
    raise NotImplementedError("Calling an abstract method.")


class _ExampleInExampleParser(_RankingDataParser):
  """Parser for Example in Example format."""

  def _decode_as_serialized_example_list(self, serialized):
    """Decodes into serialized context and examples."""
    feature_spec = {
        "serialized_context": tf.io.FixedLenFeature([1], tf.string),
        "serialized_examples": tf.io.VarLenFeature(tf.string),
    }
    features = tf.compat.v1.io.parse_example(serialized, feature_spec)
    return features["serialized_context"], tf.sparse.to_dense(
        features["serialized_examples"], default_value="")

  def parse(self, serialized):
    """See `_RankingDataParser`."""
    (serialized_context,
     serialized_list) = self._decode_as_serialized_example_list(serialized)
    # Use static batch size whenever possible.
    batch_size = serialized_context.get_shape().as_list()[0] or tf.shape(
        input=serialized_list)[0]
    cur_list_size = tf.shape(input=serialized_list)[1]
    list_size = self._list_size

    # Apply truncation or padding to align tensor shape.
    if list_size:

      def truncate_fn():
        return tf.slice(serialized_list, [0, 0], [batch_size, list_size])

      def pad_fn():
          # Create feature spec for tf.train.Example to append
          pad_spec = {}
          # Default values are 0 or an empty byte string depending on 
          # original serialized data type
          dtype_map = {tf.float32:tf.train.Feature(
                    float_list=tf.train.FloatList(value=[0.0])), 
                         tf.int32:tf.train.Feature(
                                 int64_list=tf.train.Int64List(value=[0])), 
                         tf.string:tf.train.Feature(
                                 bytes_list=tf.train.BytesList(
                                         value=[bytes('', encoding='UTF-8')]))}
          # Create the feature spec
          for key, item in self._example_feature_spec.items():
              dtype = item.dtype
              pad_spec[key] = dtype_map[dtype]
          # Make and serialize example to append
          constant_values = tf.train.Example(
                features=tf.train.Features(feature=pad_spec))
          constant_val_str = constant_values.SerializeToString()
            
          # Add serialized padding to end of list
          return tf.pad(
              tensor=serialized_list,
              paddings=[[0, 0], [0, list_size - cur_list_size]],
              constant_values=constant_val_str)

      serialized_list = tf.cond(
          pred=cur_list_size > list_size, true_fn=truncate_fn, false_fn=pad_fn)
      cur_list_size = list_size

    features = {}
    example_features = tf.compat.v1.io.parse_example(
        tf.reshape(serialized_list, [-1]), self._example_feature_spec)
    for k, v in six.iteritems(example_features):
      features[k] = utils.reshape_first_ndims(v, 1, [batch_size, cur_list_size])

    if self._context_feature_spec:
      features.update(
          tf.compat.v1.io.parse_example(
              tf.reshape(serialized_context, [batch_size]),
              self._context_feature_spec))

    return features


def parse_from_example_in_example(serialized,
                                  list_size=None,
                                  context_feature_spec=None,
                                  example_feature_spec=None):
  """Parses an ExampleInExample batch to a feature map.

  An ExampleInExample is a tf.train.Example that has two fields:
    - `serialized_context` is a scalar of bytes. The value is a serialized
      tf.train.Example that contains context features.
    - `serialized_examples` is a repeated field of bytes. The value is a list of
      serialized tf.train.Example with each representing an example that
      contains example features.

  For example:

  ```
  serialized_context_string = Serialize({
    features {
      feature {
        key: "query_length"
        value { int64_list { value: 3 } }
      }
    }
  })

  serialized_examples_string = [
    Serialize({
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
    }),

    Serialize({
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: ["learning" "to" "rank" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    })
  ]

  serialized_context_string_2 = Serialize({
    features {
      feature {
        key: "query_length"
        value { int64_list { value: 2 } }
      }
    }
  })

  serialized_examples_string_2 = [
    Serialize({
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
    }),

    Serialize({
      features {
        feature {
          key: "unigrams"
          value { bytes_list { value: ["neural" "network" } }
        }
        feature {
          key: "utility"
          value { float_list { value: 1.0 } }
        }
      }
    })
  ]

  serialized = [
    {
      serialized_context: serialized_context_string,
      serialized_examples: serialized_examples_string,
    },
    {
      serialized_context: serialized_context_string_2,
      serialized_examples: serialized_examples_string_2,
    },
  ]
  ```

  We can use arguments:

  ```
  context_feature_spec: {
    "query_length": tf.io.FixedenFeature([1], dtypes.int64),
  }
  example_feature_spec: {
    "unigrams": tf.io.VarLenFeature(dtypes.string),
    "utility": tf.io.FixedLenFeature([1], dtypes.float32),
  }
  ```

  And the expected output is:

  ```python
  {
    "unigrams": SparseTensor(
      indices=array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0],
        [1, 1, 0], [1, 1, 1]]),
      values=["tensorflow", "learning", "to", "rank", "gbdt", "neural" ,
        "network"],
      dense_shape=array([2, 2, 3])),
    "utility": [[[ 0.], [ 1.]], [[ 0.], [ 1.]]],
    "query_length": [[3], [2]],
  }
  ```

  Args:
    serialized: (Tensor) 1-D Tensor and each entry is a serialized
      `ExampleListWithContext` proto that contains context and example list.
    list_size: (int) The number of examples for each list. If specified,
      truncation or padding is applied to make 2nd dim of output Tensors aligned
      to `list_size`. Otherwise, the 2nd dim of the output Tensors is dynamic.
    context_feature_spec: (dict) A mapping from feature keys to
      `FixedLenFeature` or `VarLenFeature` values for context in
      `ExampleListWithContext` proto.
    example_feature_spec: (dict) A mapping from feature keys to
      `FixedLenFeature` or `VarLenFeature` values for examples in
      `ExampleListWithContext` proto.

  Returns:
    A mapping from feature keys to `Tensor` or `SparseTensor`.
  """
  parser = _ExampleInExampleParser(list_size, context_feature_spec,
                                   example_feature_spec)
  return parser.parse(serialized)


def _get_scalar_default_value(dtype, default_value):
  """Gets the scalar compatible default value."""
  if dtype == tf.string:
    return default_value or ""
  elif default_value is None:
    return 0
  elif isinstance(default_value, int) or isinstance(default_value, float):
    return default_value
  elif (isinstance(default_value, list) or
        isinstance(default_value, tuple)) and len(default_value) == 1:
    return default_value[0]
  else:
    raise ValueError("Only scalar or equivalent is allowed in default_value.")


class _SequenceExampleParser(_RankingDataParser):
  """Parser for SequenceExample."""

  def parse(self, serialized):
    """See `_RankingDataParser`."""
    list_size = self._list_size
    context_feature_spec = self._context_feature_spec
    example_feature_spec = self._example_feature_spec
    # Convert `FixedLenFeature` in `example_feature_spec` to
    # `FixedLenSequenceFeature` to parse the `feature_lists` in SequenceExample.
    # In addition, we collect non-trivial `default_value`s (neither "" nor 0)
    # for post-processing. This is because no `default_value` except None is
    # allowed for `FixedLenSequenceFeature`. Also, we set allow_missing=True and
    # handle the missing feature_list later.
    fixed_len_sequence_features = {}
    padding_values = {}
    non_trivial_padding_values = {}
    for k, s in six.iteritems(example_feature_spec):
      if not isinstance(s, tf.io.FixedLenFeature):
        continue
      fixed_len_sequence_features[k] = tf.io.FixedLenSequenceFeature(
          s.shape, s.dtype, allow_missing=True)
      scalar = _get_scalar_default_value(s.dtype, s.default_value)
      padding_values[k] = scalar
      if scalar:
        non_trivial_padding_values[k] = scalar

    sequence_features = example_feature_spec.copy()
    sequence_features.update(fixed_len_sequence_features)
    context, examples, sizes = tf.io.parse_sequence_example(
        serialized,
        context_features=context_feature_spec,
        sequence_features=sequence_features)

    # Reset to no trivial padding values for example features.
    for k, v in six.iteritems(non_trivial_padding_values):
      tensor = examples[k]  # [batch_size, num_frames, feature_size]
      tensor.get_shape().assert_has_rank(3)
      size = tf.reshape(sizes[k], [-1, 1, 1])  # [batch_size, 1, 1]
      rank = tf.reshape(
          tf.tile(
              tf.range(tf.shape(input=tensor)[1]), [tf.shape(input=tensor)[0]]),
          tf.shape(input=tensor))
      tensor = tf.compat.v1.where(
          tf.less(rank, tf.cast(size, tf.int32)), tensor,
          tf.fill(tf.shape(input=tensor), tf.cast(v, tensor.dtype)))
      examples[k] = tensor

    list_size_arg = list_size
    if list_size is None:
      # Use dynamic list_size. This is needed to pad missing feature_list.
      list_size_dynamic = tf.reduce_max(
          input_tensor=tf.stack(
              [tf.shape(input=t)[1] for t in six.itervalues(examples)]))
      list_size = list_size_dynamic

    # Collect features. Truncate or pad example features to normalize the tensor
    # shape: [batch_size, num_frames, ...] --> [batch_size, list_size, ...]
    features = {}
    features.update(context)
    for k, t in six.iteritems(examples):
      # Old shape: [batch_size, num_frames, ...]
      shape = tf.shape(input=t)
      ndims = t.get_shape().rank
      num_frames = shape[1]
      # New shape: [batch_size, list_size, ...]
      new_shape = tf.concat([[shape[0], list_size], shape[2:]], 0)

      def truncate_fn(t=t, ndims=ndims, new_shape=new_shape):
        """Truncates the tensor."""
        if isinstance(t, tf.sparse.SparseTensor):
          return tf.sparse.slice(t, [0] * ndims,
                                 tf.cast(new_shape, dtype=tf.int64))
        else:
          return tf.slice(t, [0] * ndims, new_shape)

      def pad_fn(k=k,
                 t=t,
                 ndims=ndims,
                 num_frames=num_frames,
                 new_shape=new_shape):
        """Pads the tensor."""
        if isinstance(t, tf.sparse.SparseTensor):
          return tf.sparse.reset_shape(t, new_shape)
        else:
          # Paddings has shape [n, 2] where n is the rank of the tensor.
          paddings = tf.stack([[0, 0], [0, list_size - num_frames]] + [[0, 0]] *
                              (ndims - 2))
          return tf.pad(
              tensor=t, paddings=paddings, constant_values=padding_values[k])

      tensor = tf.cond(
          pred=num_frames > list_size, true_fn=truncate_fn, false_fn=pad_fn)
      # Infer static shape for Tensor. Set the 2nd dim to None and set_shape
      # merges `static_shape` with the existing static shape of the thensor.
      if not isinstance(tensor, tf.sparse.SparseTensor):
        static_shape = t.get_shape().as_list()
        static_shape[1] = list_size_arg
        tensor.set_shape(static_shape)
      features[k] = tensor

    return features


def parse_from_sequence_example(serialized,
                                list_size=None,
                                context_feature_spec=None,
                                example_feature_spec=None):
  """Parses SequenceExample to feature maps.

  The `FixedLenFeature` in `example_feature_spec` is converted to
  `FixedLenSequenceFeature` to parse `feature_list` in SequenceExample. We keep
  track of the non-trivial default_values (e.g., -1 for labels) for features in
  `example_feature_spec` and use them to replace the parsing defaults of the
  SequenceExample (i.e., 0 for numbers and "" for strings). Due to this
  complexity, we only allow scalar non-trivial default values for numbers.

  When `list_size` is None, the 2nd dim of the output Tensors are not fixed and
  vary from batch to batch. When `list_size` is specified as a positive integer,
  truncation or padding is applied so that the 2nd dim of the output Tensors is
  the specified `list_size`.

  Example:
  ```
  serialized = [
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
  context_feature_spec: {
    "query_length": tf.io.FixedenFeature([1], dtypes.int64)
  }
  example_feature_spec: {
    "unigrams": tf.io.VarLenFeature(dtypes.string),
    "utility": tf.io.FixedLenFeature([1], dtypes.float32,
      default_value=[0.])
  }
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
    serialized: (Tensor) A string Tensor for a batch of serialized
      SequenceExample.
    list_size: (int) The number of frames to keep for a SequenceExample. If
      specified, truncation or padding may happen. Otherwise, the output Tensors
      have a dynamic list size.
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
  parser = _SequenceExampleParser(list_size, context_feature_spec,
                                  example_feature_spec)
  return parser.parse(serialized)


def make_parsing_fn(data_format,
                    list_size=None,
                    context_feature_spec=None,
                    example_feature_spec=None):
  """Returns a parsing fn for a standard data format.

  Args:
    data_format: (string) See RankingDataFormat.
    list_size: (int) The number of examples to keep per ranking instance. If
      specified, truncation or padding may happen. Otherwise, the output Tensors
      have a dynamic list size.
    context_feature_spec: (dict) A mapping from feature keys to
      `FixedLenFeature` or `VarLenFeature` values for context.
    example_feature_spec: (dict) A mapping from feature keys to
      `FixedLenFeature` or `VarLenFeature` values for the list of examples.

  Returns:
    A parsing function with signature parsing_fn(serialized), where serialized
    is a string Tensor representing the serialized data in the specified
    `data_format` and the function returns a feature map.
  """
  kwargs = {
      "list_size": list_size,
      "context_feature_spec": context_feature_spec,
      "example_feature_spec": example_feature_spec,
  }
  fns_dict = {
      EIE: parse_from_example_in_example,
      SEQ: parse_from_sequence_example,
  }
  if data_format in fns_dict:
    return functools.partial(fns_dict[data_format], **kwargs)
  else:
    raise ValueError("Format {} is not supported.".format(data_format))


def build_ranking_dataset_with_parsing_fn(file_pattern,
                                          parsing_fn,
                                          batch_size,
                                          reader=tf.data.TFRecordDataset,
                                          reader_args=None,
                                          num_epochs=None,
                                          shuffle=True,
                                          shuffle_buffer_size=1000,
                                          shuffle_seed=None,
                                          prefetch_buffer_size=32,
                                          reader_num_threads=10,
                                          sloppy_ordering=True,
                                          drop_final_batch=False,
                                          num_parser_threads=None):
  """Builds a ranking tf.dataset using the provided `parsing_fn`.

  Args:
    file_pattern: (str | list(str)) List of files or patterns of file paths
      containing serialized data. See `tf.gfile.Glob` for pattern rules.
    parsing_fn: (function) It has a single argument parsing_fn(serialized).
      Users can customize this for their own data formats.
    batch_size: (int) Number of records to combine in a single batch.
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
    num_parser_threads: (int) Optional number of threads to be used with
      dataset.map() when invoking parsing_fn.

  Returns:
    A dataset of `dict` elements. Each `dict` maps feature keys to
    `Tensor` or `SparseTensor` objects.
  """
  files = tf.data.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  reader_args = reader_args or []
  dataset = files.apply(
      tf.data.experimental.parallel_interleave(
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          sloppy=sloppy_ordering))

  # Extract values if tensors are stored as key-value tuples. This happens when
  # the reader is tf.data.SSTableDataset.
  if dataset.output_types == (tf.string, tf.string):
    dataset = dataset.map(lambda _, v: v)

  # Repeat and shuffle, if needed.
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)
  if shuffle:
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size, seed=shuffle_seed)
  # The drop_remainder=True allows for static inference of batch size.
  dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)

  # Parse a batch.
  dataset = dataset.map(parsing_fn, num_parallel_calls=num_parser_threads)

  # Prefetching allows for data fetching to happen on host while model runs
  # on the accelerator. When run on CPU, makes data fecthing asynchronous.
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

  return dataset


def build_ranking_dataset(file_pattern,
                          data_format,
                          batch_size,
                          context_feature_spec,
                          example_feature_spec,
                          list_size=None,
                          **kwargs):
  """Builds a ranking tf.dataset with a standard data format.

  Args:
    file_pattern: See `build_ranking_dataset_with_parsing_fn`.
    data_format: See `make_parsing_fn`.
    batch_size: See `build_ranking_dataset_with_parsing_fn`.
    context_feature_spec: See `make_parsing_fn`.
    example_feature_spec: See `make_parsing_fn`.
    list_size: See `make_parsing_fn`.
    **kwargs: The kwargs passed to `build_ranking_dataset_with_parsing_fn`.

  Returns:
    See `build_ranking_dataset_with_parsing_fn`.
  """
  parsing_fn = make_parsing_fn(data_format, list_size, context_feature_spec,
                               example_feature_spec)
  return build_ranking_dataset_with_parsing_fn(
      file_pattern, parsing_fn=parsing_fn, batch_size=batch_size, **kwargs)


def build_ranking_serving_input_receiver_fn_with_parsing_fn(
    parsing_fn, receiver_name, default_batch_size=None):
  """Returns a receiver function with the provided `parsing_fn`.

  Args:
    parsing_fn: (function) It has a single argument parsing_fn(serialized).
      Users can customize this for their own data formats.
    receiver_name: (string) The name for the reveiver Tensor that contains the
      serialized data.
    default_batch_size: (int) Number of instances expected per batch. Leave
      unset for variable batch size (recommended).

  Returns:
    A `tf.estimator.export.ServingInputReceiver` object, which packages the
    placeholders and the resulting feature Tensors together.
  """

  def _serving_input_receiver_fn():
    """Returns a serving input receiver."""
    serialized = tf.compat.v1.placeholder(
        dtype=tf.string,
        shape=[default_batch_size],
        name="input_ranking_tensor")
    receiver_tensors = {receiver_name: serialized}
    features = parsing_fn(serialized)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  return _serving_input_receiver_fn


def build_ranking_serving_input_receiver_fn(data_format,
                                            context_feature_spec,
                                            example_feature_spec,
                                            list_size=None,
                                            receiver_name="input_ranking_data",
                                            default_batch_size=None):
  """Returns a serving input receiver fn for a standard data format.

  Args:
    data_format: (string) See RankingDataFormat.
    context_feature_spec: (dict) Map from feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    example_feature_spec: (dict) Map from  feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    list_size: (int) The number of examples to keep. If specified, truncation or
      padding may happen. Otherwise, set it to None to allow dynamic list size
      (recommended).
    receiver_name: (string) The name for the receiver tensor.
    default_batch_size: (int) Number of instances expected per batch. Leave
      unset for variable batch size (recommended).

  Returns:
    A `tf.estimator.export.ServingInputReceiver` object, which packages the
    placeholders and the resulting feature Tensors together.
  """

  parsing_fn = make_parsing_fn(
      data_format,
      list_size=list_size,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec)
  return build_ranking_serving_input_receiver_fn_with_parsing_fn(
      parsing_fn, receiver_name, default_batch_size)


# Deprecated. Please use `build_ranking_dataset`.
def read_batched_sequence_example_dataset(file_pattern,
                                          batch_size,
                                          list_size,
                                          context_feature_spec,
                                          example_feature_spec,
                                          reader=tf.data.TFRecordDataset,
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
    list_size: (int) The number of frames to keep in a SequenceExample. If
      specified, truncation or padding may happen. Otherwise, set it to None to
      allow dynamic list size.
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
    feature_size], where list_size is the number of examples.
  """
  return build_ranking_dataset(
      file_pattern,
      data_format=SEQ,
      batch_size=batch_size,
      list_size=list_size,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      reader=reader,
      reader_args=reader_args,
      num_epochs=num_epochs,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle_seed=shuffle_seed,
      prefetch_buffer_size=prefetch_buffer_size,
      reader_num_threads=reader_num_threads,
      sloppy_ordering=sloppy_ordering,
      drop_final_batch=drop_final_batch)


# Deprecated. Please use `build_ranking_serving_input_receiver_fn`.
def build_sequence_example_serving_input_receiver_fn(input_size,
                                                     context_feature_spec,
                                                     example_feature_spec,
                                                     default_batch_size=None):
  """Creates a serving_input_receiver_fn for `SequenceExample` inputs.

  A string placeholder is used for inputs. Note that the context_feature_spec
  and example_feature_spec shouldn't contain weights, labels or training
  only features in general.

  Args:
    input_size: (int) The number of frames to keep in a SequenceExample. If
      specified, truncation or padding may happen. Otherwise, set it to None to
      allow dynamic list size (recommended).
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
  return build_ranking_serving_input_receiver_fn(
      SEQ,
      context_feature_spec,
      example_feature_spec,
      list_size=input_size,
      receiver_name="sequence_example",
      default_batch_size=default_batch_size)


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
    doc_list: A list of dictionaries (one per document) where each dictionary is
      a mapping from feature ID (string) to feature value (float).

  Returns:
    A tuple consisting of a dictionary (feature ID to `Tensor`s) and a label
    `Tensor`.
  """
  # Construct output variables.
  features = {}
  for fid in range(num_features):
    features[str(fid + 1)] = np.zeros([list_size, 1], dtype=np.float32)
  labels = np.ones([list_size], dtype=np.float32) * (_PADDING_LABEL)

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

    with tf.io.gfile.GFile(path, "r") as f:
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

        yield _libsvm_generate(num_features, list_size, doc_list)

        # Reset current pointer and re-initialize document list.
        cur = qid
        doc_list = [doc]

    yield _libsvm_generate(num_features, list_size, doc_list)

  return inner_generator
