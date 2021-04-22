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
"""Ranking Model utilities and classes in Keras."""

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import logging
import tensorflow as tf

from tensorflow_ranking.python.keras import feature
from tensorflow_ranking.python.keras import layers

TensorLike = tf.types.experimental.TensorLike
TensorDict = Dict[str, TensorLike]


def create_keras_model(network,
                       loss,
                       metrics,
                       optimizer,
                       size_feature_name,
                       list_size=None):
  """Creates a Functional Keras ranking model.

  A mask is inferred from size_feature_name and passed to the network, along
  with feature dictionary as inputs.

  Args:
    network: (`tfr.keras.network.RankingNetwork`) A ranking network which
      generates a list of scores.
    loss: (`tfr.keras.losses._RankingLoss`) A ranking loss.
    metrics: (list) List of ranking metrics, `tfr.keras.metrics._RankingMetric`
      instances.
    optimizer: (`tf.keras.optimizer.Optimizer`) Optimizer to minimize ranking
      loss.
    size_feature_name: (str) Name of feature for example list sizes. If not
      None, this feature name corresponds to a `tf.int32` Tensor of size
      [batch_size] corresponding to sizes of example lists. If `None`, all
      examples are treated as valid.
    list_size: (int) The list size for example features. If None, use dynamic
      list size. A fixed list size is required for TPU training.

  Returns:
    A compiled ranking Keras model, a `tf.keras.Model` instance.
  """
  # TODO: Support compatibility with TPUs.
  keras_inputs = feature.create_keras_inputs(
      context_feature_columns=network.context_feature_columns,
      example_feature_columns=network.example_feature_columns,
      size_feature_name=size_feature_name,
      list_size=list_size)

  # Create mask from sizes and list_size.
  mask = None
  if size_feature_name is not None:
    mask = feature.GenerateMask(network.example_feature_columns,
                                size_feature_name)(
                                    keras_inputs)
  logits = network(inputs=keras_inputs, mask=mask)

  ranker = tf.keras.Model(inputs=keras_inputs, outputs=logits)
  ranker.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return ranker


class AbstractModelBuilder(metaclass=abc.ABCMeta):
  """Interface to build a ranking tf.keras.Model."""

  def __init__(self, mask_feature_name: str, name: Optional[str] = None):
    """Initializes the instance.

    Args:
      mask_feature_name: name of 2D mask boolean feature.
      name: name of the Model.
    """
    self._mask_feature_name = mask_feature_name
    self._name = name
    if self._mask_feature_name is None:
      raise ValueError("mask_feature_name cannot be None.")

  @abc.abstractmethod
  def create_inputs(self) -> Tuple[TensorDict, TensorDict, tf.Tensor]:
    """Creates context and example inputs.

    Returns:
      A tuple of
      context_inputs: maps from context feature keys to Keras Input.
      example_inputs: maps from example feature keys to Keras Input.
      mask: Keras Input for the mask feature.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def preprocess(
      self,
      context_inputs: TensorDict,
      example_inputs: TensorDict,
      mask: tf.Tensor,
  ) -> Tuple[TensorDict, TensorDict]:
    """Preprocesses context and example inputs.

    Args:
      context_inputs: maps context feature keys to tf.keras.Input.
      example_inputs: maps example feature keys to tf.keras.Input.
      mask: [batch_size, list_size]-tensor of mask for valid examples.

    Returns:
      A tuple of
      context_features: maps from context feature keys to [batch_size,
        feature_dims]-tensors of preprocessed context features.
      example_features: maps from example feature keys to [batch_size,
        list_size, feature_dims]-tensors of preprocessed example features.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def score(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
      mask: tf.Tensor,
  ) -> Union[TensorLike, TensorDict]:
    """Scores all examples and returns outputs.

    Args:
      context_features: maps from context feature keys to [batch_size,
        feature_dims]-tensors of preprocessed context features.
      example_features: maps from example feature keys to [batch_size,
        list_size, feature_dims]-tensors of preprocessed example features.
      mask: [batch_size, list_size]-tensor of mask for valid examples.

    Returns:
      A [batch_size, list_size]-tensor of logits or a dict mapping task name to
      logits in the multi-task setting.
    """
    raise NotImplementedError("Calling an abstract method.")

  def build(self) -> tf.keras.Model:
    """Builds a Keras Model for Ranking Pipeline.

    Returns:
      A tf.keras.Model.
    """
    context_inputs, example_inputs, mask = self.create_inputs()
    logging.info("Context features: %s", context_inputs)
    logging.info("Example features: %s", example_inputs)
    context_features, example_features = self.preprocess(
        context_inputs, example_inputs, mask)
    outputs = self.score(context_features, example_features, mask)
    return tf.keras.Model(
        inputs=dict(
            list(context_inputs.items()) + list(example_inputs.items()) +
            [(self._mask_feature_name, mask)]),
        outputs=outputs,
        name=self._name)


class ModelBuilder(AbstractModelBuilder):
  """Builds a tf.keras.Model.

  This class implements the AbstractModelBuilder by delegating the class
  behaviors to the following implementors that can be specified by callers:

     * input_creator: A callable or a class like `InputCreator` to implement
     `create_inputs`.
     * preprocessor: A callable or a class like `Preprocessor` to implement
     `preprocess`.
     * scorer: A callable or a class like `Scorer` to implement `score`.

  Users can subclass those implementor classes and pass the objects into this
  class to build a tf.keras.Model.
  """

  def __init__(
      self,
      input_creator: Callable[[], Tuple[TensorDict, TensorDict]],
      preprocessor: Callable[[TensorDict, TensorDict, tf.Tensor],
                             Tuple[TensorDict, TensorDict]],
      scorer: Callable[[TensorDict, TensorDict, tf.Tensor], Union[TensorLike,
                                                                  TensorDict]],
      mask_feature_name: str,
      name: Optional[str] = None,
  ):
    """Initializes the instance."""
    super().__init__(mask_feature_name, name)
    self._input_creator = input_creator
    self._preprocessor = preprocessor
    self._scorer = scorer

  def create_inputs(self) -> Tuple[TensorDict, TensorDict, tf.Tensor]:
    """See `AbstractModelBuilder`."""
    context_inputs, example_inputs = self._input_creator()
    mask = tf.keras.Input(
        name=self._mask_feature_name, shape=(None,), dtype=tf.bool)
    return context_inputs, example_inputs, mask

  def preprocess(
      self,
      context_inputs: TensorDict,
      example_inputs: TensorDict,
      mask: tf.Tensor,
  ) -> Tuple[TensorDict, TensorDict]:
    """See `AbstractModelBuilder`."""
    return self._preprocessor(context_inputs, example_inputs, mask)

  def score(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
      mask: tf.Tensor,
  ) -> Union[TensorLike, TensorDict]:
    """See `AbstractModelBuilder`."""
    return self._scorer(context_features, example_features, mask)


class InputCreator(metaclass=abc.ABCMeta):
  """Interface for input creator."""

  @abc.abstractmethod
  def __call__(self) -> Tuple[TensorDict, TensorDict]:
    raise NotImplementedError("Calling an abstract method.")


class FeatureSpecInputCreator(InputCreator):
  """InputCreator with feature specs."""

  def __init__(
      self,
      context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
      example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
  ):
    self._context_feature_spec = context_feature_spec
    self._example_feature_spec = example_feature_spec

  def __call__(self) -> Tuple[TensorDict, TensorDict]:
    """See `InputCreator`."""

    def get_keras_input(feature_spec, name, is_example=False):
      if isinstance(feature_spec, tf.io.FixedLenFeature):
        return tf.keras.Input(
            shape=(None,) + tuple(feature_spec.shape)
            if is_example else tuple(feature_spec.shape),
            dtype=feature_spec.dtype,
            name=name)
      elif isinstance(feature_spec, tf.io.VarLenFeature):
        return tf.keras.Input(
            shape=(None, 1) if is_example else (1),
            dtype=feature_spec.dtype,
            name=name,
            sparse=True)
      elif isinstance(feature_spec, tf.io.RaggedFeature):
        return tf.keras.Input(
            shape=(None,) *
            (len(feature_spec.partitions) + 2) if is_example else
            (None,) * (len(feature_spec.partitions) + 1),
            dtype=feature_spec.dtype,
            name=name,
            ragged=True)
      else:
        raise ValueError("{} is not supported.".format(feature_spec))

    context_inputs = {
        name: get_keras_input(spec, name)
        for name, spec in self._context_feature_spec.items()
    }
    example_inputs = {
        name: get_keras_input(spec, name, is_example=True)
        for name, spec in self._example_feature_spec.items()
    }
    return context_inputs, example_inputs


class TypeSpecInputCreator(InputCreator):
  """InputCreator with tensor type specs."""

  def __init__(
      self,
      type_spec: Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]],
      context_feature_names: Optional[List[str]],
      example_feature_names: Optional[List[str]],
  ):
    self._type_spec = type_spec
    self._context_feature_names = context_feature_names or []
    self._example_feature_names = example_feature_names or []

  def __call__(self) -> Tuple[TensorDict, TensorDict]:
    """See `InputCreator`."""

    def get_keras_input(type_spec, name=None):
      """Returns a keras.Input."""
      return tf.keras.Input(
          shape=type_spec.shape[1:],
          dtype=type_spec.dtype,
          ragged=isinstance(type_spec, tf.RaggedTensorSpec),
          name=name)

    context_inputs, example_inputs = {}, {}
    for name, spec in self._type_spec.items():
      k_input = get_keras_input(spec, name)
      if name in self._context_feature_names:
        context_inputs[name] = k_input
      elif name in self._example_feature_names:
        example_inputs[name] = k_input
    return context_inputs, example_inputs


class Preprocessor(metaclass=abc.ABCMeta):
  """Interface for feature preprocessing."""

  @abc.abstractmethod
  def __call__(
      self,
      context_inputs: TensorDict,
      example_inputs: TensorDict,
      mask: tf.Tensor,
  ) -> Tuple[TensorDict, TensorDict]:
    raise NotImplementedError("Calling an abstract method.")


class PreprocessorWithSpec(Preprocessor):
  """Preprocessing inputs with provided spec.

  Transformation including KPL or customized transformation like log1p can be
  defined and passed in `preprocess_spec`:
      preprocess_spec = {
          **{name: lambda t: tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
             for name in example_feature_spec.keys()},
          **{name: tf.reduce_mean(
              tf.keras.layers.Embedding(input_dim=10, output_dim=4)(x), axis=-2)
             for name in context_feature_spec.keys()}
      }
  """

  def __init__(
      self,
      preprocess_spec: Optional[Dict[str, Callable[[Any], Any]]] = None,
      default_value_spec: Optional[Dict[str, float]] = None,
  ):
    """Initializer.

    Args:
      preprocess_spec: maps a feature name to a callable to preprocess a
        feature. Only include those features that need preprocessing.
      default_value_spec: maps a feature name to a default value to convert a
        RaggedTensor to Tensor. Default to 0. if not specified.
    """
    self._preprocess_spec = preprocess_spec or {}
    self._default_value_spec = default_value_spec or {}

  def __call__(
      self,
      context_inputs: TensorDict,
      example_inputs: TensorDict,
      mask: tf.Tensor,
  ) -> Tuple[TensorDict, TensorDict]:
    """See `Preprocessor`."""
    list_size = tf.shape(mask)[1]

    def apply_preprocess(key, value, is_example=False):
      """Applies the preprocessing spec and convert to tf.Tensor."""
      if key in self._preprocess_spec:
        value = self._preprocess_spec[key](value)
      if isinstance(value, tf.RaggedTensor):
        default_value = self._default_value_spec.get(key, 0.)
        s = value.bounding_shape()
        new_shape = tf.concat([s[:1], [list_size], s[2:]],
                              axis=0) if is_example else s
        return value.to_tensor(default_value, shape=new_shape)
      else:
        return value

    context_features = {
        key: apply_preprocess(key, value)
        for key, value in context_inputs.items()
    }
    example_features = {
        key: apply_preprocess(key, value, True)
        for key, value in example_inputs.items()
    }
    return context_features, example_features


class Scorer(metaclass=abc.ABCMeta):
  """Interface for scorer."""

  @abc.abstractmethod
  def __call__(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
      mask: tf.Tensor,
  ) -> Union[TensorLike, TensorDict]:
    """Scores all examples given context and returns logits.

    Args:
      context_features: maps from context feature keys to [batch_size,
        feature_dims]-tensors of preprocessed context features.
      example_features: maps from example feature keys to [batch_size,
        list_size, feature_dims]-tensors of preprocessed example features.
      mask: [batch_size, list_size]-tensor of mask for valid examples.

    Returns:
      A [batch_size, list_size]-tensor of logits or a dict mapping task name to
      logits in the multi-task setting.
    """
    raise NotImplementedError("Calling an abstract method.")


class UnivariateScorer(Scorer, metaclass=abc.ABCMeta):
  """Interface for univariate scorer."""

  @abc.abstractmethod
  def _score_flattened(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
  ) -> Union[tf.Tensor, TensorDict]:
    """Computes the flattened logits."""
    raise NotImplementedError("Calling an abstract method.")

  def __call__(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
      mask: tf.Tensor,
  ) -> Union[tf.Tensor, TensorDict]:
    """Scores all examples and returns logits."""
    (flattened_context_features,
     flattened_example_features) = layers.FlattenList()(
         inputs=(context_features, example_features, mask))

    flattened_logits = self._score_flattened(flattened_context_features,
                                             flattened_example_features)

    # Handle a dict of logits for the multi-task setting.
    if isinstance(flattened_logits, dict):
      logits = {
          k: layers.RestoreList(name=k)(inputs=(v, mask))
          for k, v in flattened_logits.items()
      }
    else:
      logits = layers.RestoreList()(inputs=(flattened_logits, mask))
    return logits


class DNNScorer(UnivariateScorer):
  """Univariate scorer using DNN."""

  def __init__(self, **dnn_kwargs):
    self._dnn_kwargs = dnn_kwargs

  def _score_flattened(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
  ) -> tf.Tensor:
    """See `UnivariateScorer`."""
    context_input_layer = [
        tf.keras.layers.Flatten()(context_features[name])
        for name in sorted(context_features)
    ]
    example_input_layer = [
        tf.keras.layers.Flatten()(example_features[name])
        for name in sorted(example_features)
    ]
    input_layer = tf.concat(context_input_layer + example_input_layer, 1)
    flattened_logits = layers.create_tower(**self._dnn_kwargs)(input_layer)

    return flattened_logits


class GAMScorer(UnivariateScorer):
  """Univariate scorer using GAM."""

  def __init__(self, **gam_kwargs):
    self._gam_kwargs = gam_kwargs

  def _score_flattened(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
  ) -> tf.Tensor:
    """See `UnivariateScorer`."""
    context_inputs = [
        tf.keras.layers.Flatten()(context_features[name])
        for name in sorted(context_features)
    ]
    example_inputs = [
        tf.keras.layers.Flatten()(example_features[name])
        for name in sorted(example_features)
    ]
    gam_kwargs = self._gam_kwargs
    # TODO: These parameters may be inferred from the call inputs.
    gam_kwargs.update({
        "example_feature_num": len(example_inputs),
        "context_feature_num": len(context_inputs),
    })
    flattened_logits, _, _ = layers.GAMLayer(**gam_kwargs)(
        inputs=(example_inputs, context_inputs))
    return flattened_logits
