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
"""Ranking model utilities and classes in tfr.keras."""

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
  """Interface to build a `tf.keras.Model` for ranking.

  The `AbstractModelBuilder` serves as the interface between model building and
  training. The training pipeline just calls the `build()` method to get the
  model constructed in the strategy scope used in the training pipeline, so for
  all variables in the model, optimizers, and metrics. See `ModelFitPipeline` in
  `pipeline.py` for example.

  The `build()` method is to be implemented in a subclass. The simplest example
  is just to define everything inside the build function when you define a
  tf.keras.Model.

  ```python
  class MyModelBuilder(AbstractModelBuilder):

    def build(self) -> tf.keras.Model:
      inputs = ...
      outputs = ...
      return tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  The `MyModelBuilder` should work with `ModelFitPipeline`. To make the model
  building more structured for ranking problems, we also define subclasses like
  `ModelBuilderWithMask` in the following.
  """

  @abc.abstractmethod
  def build(self) -> tf.keras.Model:
    """The build method to be implemented by a subclass."""
    raise NotImplementedError("Calling an abstract method.")


class ModelBuilderWithMask(AbstractModelBuilder, metaclass=abc.ABCMeta):
  """Interface to build a `tf.keras.Model` for ranking with a mask Tensor.

  The `ModelBuilderWithMask` class is an abstract class to build a ranking model
  based on dense Tensors and a mask Tensor to indicate the padded ones.
  All the boilerplate codes related to constructing a `tf.keras.Model` are
  integrated in the ModelBuilder class.

  To be implemented by subclasses:

    * `create_inputs()`: Contains the logic to create `tf.keras.Input` for
    context and example inputs and mask for valid list items.
    * `preprocess()`: Contains the logic to preprocess context and example
    inputs.
    * `score()`: Contains the logic to score examples in list and return
    outputs.

  Example subclass implementation:

  ```python
  class SimpleModelBuilder(ModelBuilderWithMask):

    def __init__(self, context_feature_spec, example_feature_spec,
                 mask_feature_name, name=None):
      self._context_feature_spec = context_feature_spec
      self._example_feature_spec = example_feature_spec
      self._mask_feature_name = mask_feature_name
      self._name = name

    def create_inputs(self):
      context_inputs = {
          name: tf.keras.Input(
              shape=tuple(spec.shape),
              name=name,
              dtype=spec.dtype
          ) for name, spec in self._context_feature_spec.items()
      }
      example_inputs = {
          name: tf.keras.Input(
              shape=(None,) + tuple(spec.shape),
              name=name,
              dtype=spec.dtype
          ) for name, spec in self._example_feature_spec.items()
      }
      mask = tf.keras.Input(
          name=self._mask_feature_name, shape=(None,), dtype=tf.bool)
      return context_inputs, example_inputs, mask

    def preprocess(self, context_inputs, example_inputs, mask):
      context_features = {
          name: tf.math.log1p(
              tf.abs(tensor)) for name, tensor in context_inputs.items()
      }
      example_features = {
          name: tf.math.log1p(
              tf.abs(tensor)) for name, tensor in example_inputs.items()
      }
      return context_features, example_features

    def score(self, context_features, example_features, mask):
      x = tf.concat([tensor for tensor in example_features.values()], -1)
      return tf.keras.layers.Dense(1)(x)
  ```
  """

  def __init__(self, mask_feature_name: str, name: Optional[str] = None):
    """Initializes the instance.

    Args:
      mask_feature_name: name of 2D mask boolean feature.
      name: (optional) name of the Model.
    """
    self._mask_feature_name = mask_feature_name
    self._name = name
    if self._mask_feature_name is None:
      raise ValueError("mask_feature_name cannot be None.")

  @abc.abstractmethod
  def create_inputs(self) -> Tuple[TensorDict, TensorDict, tf.Tensor]:
    """Creates context and example inputs.

    Example usage:

    ```python
    model_builder = SimpleModelBuilder(
        {},
        {"example_feature_1": tf.io.FixedLenFeature(
            shape=(1,), dtype=tf.float32, default_value=0.0)},
        "list_mask", "model_builder")
    context_inputs, example_inputs, mask = model_builder.create_inputs()
    ```

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

    Example usage:

    ```python
    model_builder = SimpleModelBuilder(
        {},
        {"example_feature_1": tf.io.FixedLenFeature(
            shape=(1,), dtype=tf.float32, default_value=0.0)},
        "list_mask", "model_builder")
    context_inputs, example_inputs, mask = model_builder.create_inputs()
    context_features, example_features = model_builder.preprocess(
        context_inputs, example_inputs, mask)
    ```

    Args:
      context_inputs: maps context feature keys to `tf.keras.Input`.
      example_inputs: maps example feature keys to `tf.keras.Input`.
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

    Example usage:

    ```python
    model_builder = SimpleModelBuilder(
        {},
        {"example_feature_1": tf.io.FixedLenFeature(
            shape=(1,), dtype=tf.float32, default_value=0.0)},
        "list_mask", "model_builder")
    context_inputs, example_inputs, mask = model_builder.create_inputs()
    context_features, example_features = model_builder.preprocess(
        context_inputs, example_inputs, mask)
    scores = model_builder.score(context_features, example_features)
    ```

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

    Example usage:

    ```python
    model_builder = SimpleModelBuilder(
        {},
        {"example_feature_1": tf.io.FixedLenFeature(
            shape=(1,), dtype=tf.float32, default_value=0.0)},
        "list_mask", "model_builder")
    model = model_builder.build()
    ```

    Returns:
      A `tf.keras.Model`.
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


class ModelBuilder(ModelBuilderWithMask):
  """Builds a `tf.keras.Model`.

  This class implements the `ModelBuilderWithMask` by delegating the class
  behaviors to the following implementors that can be specified by callers:

     * input_creator: A callable or a class like `InputCreator` to implement
     `create_inputs`.
     * preprocessor: A callable or a class like `Preprocessor` to implement
     `preprocess`.
     * scorer: A callable or a class like `Scorer` to implement `score`.

  Users can subclass those implementor classes and pass the objects into this
  class to build a `tf.keras.Model`.

  Example usage:

  ```python
  model_builder = ModelBuilder(
      input_creator=FeatureSpecInputCreator(
          {},
          {"example_feature_1": tf.io.FixedLenFeature(
              shape=(1,), dtype=tf.float32, default_value=0.0)}),
      preprocessor=PreprocessorWithSpec(),
      scorer=DNNScorer(hidden_layer_dims=[16]),
      mask_feature_name="list_mask",
      name="model_builder")
  ```
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
    """Initializes the instance.

    Args:
      input_creator: A callable or a class like `InputCreator` to implement
        `create_inputs`.
      preprocessor: A callable or a class like `Preprocessor` to implement
        `preprocess`.
      scorer: A callable or a class like `Scorer` to implement `score`.
      mask_feature_name: name of 2D mask boolean feature.
      name: (optional) name of the Model.
    """
    super().__init__(mask_feature_name, name)
    self._input_creator = input_creator
    self._preprocessor = preprocessor
    self._scorer = scorer

  def create_inputs(self) -> Tuple[TensorDict, TensorDict, tf.Tensor]:
    """See `ModelBuilderWithMask`."""
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
    """See `ModelBuilderWithMask`."""
    return self._preprocessor(context_inputs, example_inputs, mask)

  def score(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
      mask: tf.Tensor,
  ) -> Union[TensorLike, TensorDict]:
    """See `ModelBuilderWithMask`."""
    return self._scorer(context_features, example_features, mask)


class InputCreator(metaclass=abc.ABCMeta):
  """Interface for input creator.

  The `InputCreator` class is an abstract class to implement `create_inputs` in
  `ModelBuilder` in tfr.keras.

  To be implemented by subclasses:

    * `__call__()`: Contains the logic to create `tf.keras.Input` for context
    and example inputs.

  Example subclass implementation:

  ```python
  class SimpleInputCreator(InputCreator):

    def __call__(self):
      return {}, {"example_feature_1": tf.keras.Input((1,), dtype=tf.float32)}
  ```
  """

  @abc.abstractmethod
  def __call__(self) -> Tuple[TensorDict, TensorDict]:
    """Invokes the `InputCreator` instance.

    Returns:
      A tuple of two dicts which map the context and example feature keys to
        the corresponding `tf.keras.Input`.
    """
    raise NotImplementedError("Calling an abstract method.")


class FeatureSpecInputCreator(InputCreator):
  """InputCreator with feature specs.

  Example usage:

  ```python
  input_creator=FeatureSpecInputCreator(
      {},
      {"example_feature_1": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)})
  ```
  """

  def __init__(
      self,
      context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
      example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
  ):
    """Initializes the instance.

    Args:
      context_feature_spec: A dict maps the context feature keys to the
        corresponding context feature specs.
      example_feature_spec: A dict maps the example feature keys to the
        corresponding example feature specs.
    """
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
  """InputCreator with tensor type specs.

  Example usage:

  ```python
  input_creator=TypeSpecInputCreator(
      {"example_feature_1": tf.TensorSpec(shape=[None, 1], dtype=tf.float32)},
      example_feature_names=["example_feature_1"])
  ```
  """

  def __init__(
      self,
      type_spec: Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]],
      context_feature_names: Optional[List[str]] = None,
      example_feature_names: Optional[List[str]] = None,
  ):
    """Initializes the instance.

    Args:
      type_spec: A dict maps the context and example feature keys to the
        corresponding context and example type specs.
      context_feature_names: A list of context feature keys.
      example_feature_names: A list of example feature keys.
    """
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
  """Interface for feature preprocessing.

  The `Preprocessor` class is an abstract class to implement `preprocess` in
  `ModelBuilder` in tfr.keras.

  To be implemented by subclasses:

    * `__call__()`: Contains the logic to preprocess context and example inputs.

  Example subclass implementation:

  ```python
  class SimplePreprocessor(Preprocessor):

    def __call__(self, context_inputs, example_inputs, mask):
      context_features = {
          name: tf.math.log1p(
              tf.abs(tensor)) for name, tensor in context_inputs.items()
      }
      example_features = {
          name: tf.math.log1p(
              tf.abs(tensor)) for name, tensor in example_inputs.items()
      }
      return context_features, example_features
  ```
  """

  @abc.abstractmethod
  def __call__(
      self,
      context_inputs: TensorDict,
      example_inputs: TensorDict,
      mask: tf.Tensor,
  ) -> Tuple[TensorDict, TensorDict]:
    """Invokes the `Preprocessor` instance.

    Args:
      context_inputs: maps context feature keys to `tf.keras.Input`.
      example_inputs: maps example feature keys to `tf.keras.Input`.
      mask: [batch_size, list_size]-tensor of mask for valid examples.

    Returns:
      A tuple of two dicts which map the context and example feature keys to
        the corresponding `tf.Tensor`s.
    """
    raise NotImplementedError("Calling an abstract method.")


class PreprocessorWithSpec(Preprocessor):
  """Preprocessing inputs with provided spec.

  Transformation including KPL or customized transformation like log1p can be
  defined and passed in `preprocess_spec` with the following example usage:

  ```python
  preprocess_spec = {
      **{name: lambda t: tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
         for name in example_feature_spec.keys()},
      **{name: tf.reduce_mean(
          tf.keras.layers.Embedding(input_dim=10, output_dim=4)(x), axis=-2)
         for name in context_feature_spec.keys()}
  }
  preprocessor = PreprocessorWithSpec(preprocess_spec)
  ```
  """

  def __init__(
      self,
      preprocess_spec: Optional[Dict[str, Callable[[Any], Any]]] = None,
      default_value_spec: Optional[Dict[str, float]] = None,
  ):
    """Initializes the instance.

    Args:
      preprocess_spec: maps a feature key to a callable to preprocess a feature.
        Only include those features that need preprocessing.
      default_value_spec: maps a feature key to a default value to convert a
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
  """Interface for scorer.

  The `Scorer` class is an abstract class to implement `score` in `ModelBuilder`
  in tfr.keras.

  To be implemented by subclasses:

    * `__call__()`: Contains the logic to score based on the context and example
    features.

  Example subclass implementation:

  ```python
  class SimpleScorer(Scorer):

    def __call__(self, context_features, example_features, mask):
      x = tf.concat([tensor for tensor in example_features.values()], -1)
      return tf.keras.layers.Dense(1)(x)
  ```
  """

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
  """Interface for univariate scorer.

  The `UnivariateScorer` class is an abstract class to implement `score` in
  `ModelBuilder` in tfr.keras with a univariate scoring function.

  To be implemented by subclasses:

    * `_score_flattened()`: Contains the logic to do the univariate scoring on
    flattened context and example features.

  Example subclass implementation:

  ```python
  class SimpleUnivariateScorer(UnivariateScorer):

    def _score_flattened(self, context_features, example_features):
      x = tf.concat([tensor for tensor in example_features.values()], -1)
      return tf.keras.layers.Dense(1)(x)
  ```
  """

  @abc.abstractmethod
  def _score_flattened(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
  ) -> Union[tf.Tensor, TensorDict]:
    """Computes the flattened logits.

    Args:
      context_features: maps from context feature keys to [batch_size *
        list_size, feature_dims]-tensors of preprocessed context features.
      example_features: maps from example feature keys to [batch_size *
        list_size, feature_dims]-tensors of preprocessed example features.

    Returns:
      A tf.Tensor of size [batch_size * list_size, 1] or a dict maps output
        names to tf.Tensors of size [batch_size * list_size, 1].
    """
    raise NotImplementedError("Calling an abstract method.")

  def __call__(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
      mask: tf.Tensor,
  ) -> Union[tf.Tensor, TensorDict]:
    """See `Scorer`."""
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
  """Univariate scorer using DNN.

  Example usage:

  ```python
  scorer=DNNScorer(hidden_layer_dims=[16])
  ```
  """

  def __init__(self, **dnn_kwargs):
    """Initializes the instance.

    Args:
      **dnn_kwargs: A dict of keyward arguments for dense neural network layers.
        Please see `tfr.keras.layers.create_tower` for specific list of keyword
        arguments.
    """
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
  """Univariate scorer using GAM.

  The scorer implements Neural Generalized Additive Ranking Model, which is an
  additive ranking model.
  See the [paper](https://arxiv.org/abs/2005.02553) for more details.

  Example usage:

  ```python
  scorer=GAMScorer(hidden_layer_dims=[16])
  ```
  """

  def __init__(self, **gam_kwargs):
    """Initializes the instance.

    Args:
      **gam_kwargs: A dict of keyward arguments for GAM layers. Please see
        `tfr.keras.layers.GAMlayer` for specific list of keyword arguments.
    """
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
