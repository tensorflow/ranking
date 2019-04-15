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

"""Feature transformations for ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.estimator import model_fn

from tensorflow_ranking.python import utils


def make_identity_transform_fn(context_feature_names):
  """Returns transform fn that split the features.

    The make_identity_transform_fn generates a transform_fn which handles only
    non-prefixed features. The per-example features need to have shape
    [batch_size, input_size, ...] and the context features need to have shape
    [batch_size, ...].

  Args:
    context_feature_names: A list of strings representing the context feature
      names.

  Returns:
    An identity transform function that splits into context and per example
    features.
  """

  def _transform_fn(features, mode):
    """Splits the features into context and per-example features."""
    del mode
    context_features = {
        name: feature
        for name, feature in six.iteritems(features)
        if name in context_feature_names
    }

    per_example_features = {
        name: feature
        for name, feature in six.iteritems(features)
        if name not in context_feature_names
    }

    return context_features, per_example_features

  return _transform_fn


def encode_features(features,
                    feature_columns,
                    mode=model_fn.ModeKeys.TRAIN,
                    scope=None):
  """Returns dense tensors from features using feature columns.

  This function encodes the feature column transformation on the 'raw'
  `features`.


  Args:
    features: (dict) mapping feature names to feature values, possibly obtained
      from input_fn.
    feature_columns: (list)  list of feature columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    (dict) A mapping from columns to dense tensors.
  """
  # Having scope here for backward compatibility.
  del scope
  trainable = (mode == model_fn.ModeKeys.TRAIN)
  cols_to_tensors = {}

  if hasattr(feature_column_lib, "is_feature_column_v2"
            ) and feature_column_lib.is_feature_column_v2(feature_columns):
    dense_layer = feature_column_lib.DenseFeatures(
        feature_columns=feature_columns,
        name="encoding_layer",
        trainable=trainable)
    dense_layer(features, cols_to_output_tensors=cols_to_tensors)
  else:
    feature_column.input_layer(
        features=features,
        feature_columns=feature_columns,
        trainable=trainable,
        cols_to_output_tensors=cols_to_tensors)

  return cols_to_tensors


def encode_listwise_features(features,
                             input_size,
                             context_feature_columns,
                             example_feature_columns,
                             mode=model_fn.ModeKeys.TRAIN,
                             scope=None):
  """Returns dense tensors from features using feature columns.

  Args:
    features: (dict) mapping feature names (str) to feature values (`tf.Tensor`
      or `tf.SparseTensor`), possibly obtained from input_fn. For context
      features, the tensors are 2-D, while for example features the tensors are
      3-D.
    input_size: (int) number of examples per query. This is the size of second
      dimension of the Tensor corresponding to one of the example feature
      columns.
    context_feature_columns: (dict) context feature names to columns.
    example_feature_columns: (dict) example feature names to columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    context_features: (dict) A mapping from context feature names to dense
    2-D tensors of shape [batch_size, ...].
    example_features: (dict) A mapping frome example feature names to dense
    3-D tensors of shape [batch_size, input_size, ...].

  Raises:
    ValueError: If `input size` is not equal to 2nd dimension of example
    tensors.
  """
  context_features = {}
  if context_feature_columns:
    context_cols_to_tensors = encode_features(
        features, context_feature_columns.values(), mode=mode, scope=scope)
    context_features = {
        name: context_cols_to_tensors[col]
        for name, col in six.iteritems(context_feature_columns)
    }

  example_features = {}
  if example_feature_columns:
    # Reshape [batch_size, input_size] to [batch * input_size] so that
    # features are encoded.
    batch_size = None
    reshaped_features = {}
    for name in example_feature_columns:
      if name not in features:
        continue
      batch_size = array_ops.shape(features[name])[0]
      try:
        reshaped_features[name] = utils.reshape_first_ndims(
            features[name], 2, [batch_size * input_size])
      except:
        raise ValueError(
            "2nd dimesion of tensor must be equal to input size: {}, "
            "but found feature {} with shape {}.".format(
                input_size, name, features[name].get_shape()))
    example_cols_to_tensors = encode_features(
        reshaped_features,
        example_feature_columns.values(),
        mode=mode,
        scope=scope)
    example_features = {
        name: utils.reshape_first_ndims(example_cols_to_tensors[col], 1,
                                        [batch_size, input_size])
        for name, col in six.iteritems(example_feature_columns)
    }

  return context_features, example_features


def encode_pointwise_features(features,
                              context_feature_columns,
                              example_feature_columns,
                              mode=model_fn.ModeKeys.PREDICT,
                              scope=None):
  """Returns dense tensors from pointwise features using feature columns.

  Args:
    features: (dict) mapping feature names to 2-D tensors, possibly obtained
      from input_fn.
    context_feature_columns: (dict) context feature names to columns.
    example_feature_columns: (dict) example feature names to columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    context_features: (dict) A mapping from context feature names to dense
    2-D tensors of shape [batch_size, ...].
    example_features: (dict) A mapping frome example feature names to dense
    3-D tensors of shape [batch_size, 1, ...].
  """
  context_features = {}
  if context_feature_columns:
    context_cols_to_tensors = encode_features(
        features, context_feature_columns.values(), mode=mode, scope=scope)
    context_features = {
        name: context_cols_to_tensors[col]
        for name, col in six.iteritems(context_feature_columns)
    }

  example_features = {}
  if example_feature_columns:
    # Handles the case when tf.Example is used as input during serving.
    example_cols_to_tensors = encode_features(
        features, example_feature_columns.values(), mode=mode, scope=scope)
    example_features = {
        name: array_ops.expand_dims(example_cols_to_tensors[col], 1)
        for name, col in six.iteritems(example_feature_columns)
    }

  return context_features, example_features
