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

"""Feature transformations for ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow.python.feature_column import feature_column_lib
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
                    mode=tf.estimator.ModeKeys.TRAIN,
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
  trainable = (mode == tf.estimator.ModeKeys.TRAIN)
  cols_to_tensors = {}

  # TODO: Ensure only v2 Feature Columns are used.
  if hasattr(feature_column_lib, "is_feature_column_v2"
            ) and feature_column_lib.is_feature_column_v2(feature_columns):
    dense_layer = feature_column_lib.DenseFeatures(
        feature_columns=feature_columns,
        name="encoding_layer",
        trainable=trainable)
    dense_layer(features, cols_to_output_tensors=cols_to_tensors)
  else:
    tf.compat.v1.feature_column.input_layer(
        features=features,
        feature_columns=feature_columns,
        trainable=trainable,
        cols_to_output_tensors=cols_to_tensors)

  return cols_to_tensors


def encode_listwise_features(features,
                             context_feature_columns,
                             example_feature_columns,
                             input_size=None,
                             mode=tf.estimator.ModeKeys.TRAIN,
                             scope=None):
  """Returns dense tensors from features using feature columns.

  Args:
    features: (dict) mapping feature names (str) to feature values (`tf.Tensor`
      or `tf.SparseTensor`), possibly obtained from input_fn. For context
      features, the tensors are 2-D, while for example features the tensors are
      3-D.
    context_feature_columns: (dict) context feature names to columns.
    example_feature_columns: (dict) example feature names to columns.
    input_size: (int) [DEPRECATED: Use without this argument.] number of
      examples per query. If this is None, input_size is inferred as the size
      of second dimension of the Tensor corresponding to one of the example
      feature columns.
    mode: (`estimator.ModeKeys`) Specifies if this is training, evaluation or
      inference. See `ModeKeys`.
    scope: (str) variable scope for the per column input layers.

  Returns:
    context_features: (dict) A mapping from context feature names to dense
    2-D tensors of shape [batch_size, ...].
    example_features: (dict) A mapping from example feature names to dense
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

  # Compute example_features. Note that the keys in `example_feature_columns`
  # dict can be different from the keys in the `features` dict. We only need to
  # reshape the per-example tensors in `features`. To obtain the keys for
  # per-example features, we use the parsing feature specs.
  example_features = {}
  if example_feature_columns:
    if feature_column_lib.is_feature_column_v2(
        example_feature_columns.values()):
      example_specs = tf.compat.v2.feature_column.make_parse_example_spec(
          example_feature_columns.values())
    else:
      example_specs = tf.compat.v1.feature_column.make_parse_example_spec(
          example_feature_columns.values())
    example_name = next(six.iterkeys(example_specs))
    batch_size = tf.shape(input=features[example_name])[0]
    if input_size is None:
      input_size = tf.shape(input=features[example_name])[1]
    # Reshape [batch_size, input_size] to [batch * input_size] so that
    # features are encoded.
    reshaped_features = {}
    for name in example_specs:
      if name not in features:
        tf.compat.v1.logging.warn("Feature {} is not found.".format(name))
        continue
      try:
        reshaped_features[name] = utils.reshape_first_ndims(
            features[name], 2, [batch_size * input_size])
      except:
        raise ValueError(
            "2nd dimension of tensor must be equal to input size: {}, "
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
                              mode=tf.estimator.ModeKeys.PREDICT,
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
    example_features: (dict) A mapping from example feature names to dense
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
        name: tf.expand_dims(example_cols_to_tensors[col], 1)
        for name, col in six.iteritems(example_feature_columns)
    }

  return context_features, example_features
