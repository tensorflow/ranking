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
"""SavedModel utilities for TF-Ranking."""

import functools
from typing import Callable, Dict, Union

import tensorflow as tf
from tensorflow_ranking.python import data


class Signatures(tf.Module):
  """Defines signatures to support regress and predict serving.

  This wraps the trained Keras model in two serving functions that can be saved
  with `tf.saved_model.save` or `model.save`, and loaded with corresponding
  signature names. The regress serving signature takes a batch of serialized
  `tf.Example`s as input, whereas the predict serving signature takes a batch of
  serialized `ExampleListWithContext` as input.

  Example usage:

  A ranking model can be saved with signatures as follows:

  ```python
  tf.saved_model.save(model, path, signatures=Signatures(model, ...)())
  ```

  For regress serving, scores can be generated using `REGRESS` signature as
  follows:

  ```python
  loaded_model = tf.saved_model.load(path)
  predictor = loaded_model.signatures[tf.saved_model.REGRESS_METHOD_NAME]
  scores = predictor(serialized_examples)[tf.saved_model.REGRESS_OUTPUTS]
  ```

  For predict serving, scores can be generated using `PREDICT` signature as
  follows:

  ```python
  loaded_model = tf.saved_model.load(path)
  predictor = loaded_model.signatures[tf.saved_model.PREDICT_METHOD_NAME]
  scores = predictor(serialized_elwcs)[tf.saved_model.PREDICT_OUTPUTS]
  ```
  """

  def __init__(self, model: tf.keras.Model,
               context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                                     tf.io.RaggedFeature]],
               example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                                     tf.io.RaggedFeature]],
               mask_feature_name: str):
    """Constructor.

    Args:
      model: A Keras ranking model.
      context_feature_spec: (dict) A mapping from feature keys to
        `FixedLenFeature` or `RaggedFeature` values for context in
        `tensorflow.serving.ExampleListWithContext` proto.
      example_feature_spec: (dict) A mapping from feature keys to
        `FixedLenFeature` or `Ragged` values for examples in
        `tensorflow.serving.ExampleListWithContext` proto.
      mask_feature_name: (str) Name of feature for example list masks.
    """
    super().__init__()
    self._model = model
    self._context_feature_spec = context_feature_spec
    self._example_feature_spec = example_feature_spec
    self._mask_feature_name = mask_feature_name

  def normalize_outputs(
      self, default_key: str,
      outputs: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
    """Returns a dict of Tensors for outputs.

    Args:
      default_key: If outputs is a Tensor, use the default_key to make a dict.
      outputs: outputs to be normalized.

    Returns:
      A dict maps from str-like key(s) to Tensor(s).

    Raises:
      TypeError if outputs is not a Tensor nor a dict.
    """
    if isinstance(outputs, tf.Tensor):
      return {default_key: outputs}
    elif isinstance(outputs, dict):
      return outputs
    else:
      raise TypeError(
          "Model outputs need to be either a tensor or a dict of tensors.")

  def predict_tf_function(self) -> Callable[[tf.Tensor], Dict[str, tf.Tensor]]:
    """Makes a tensorflow function for `predict`."""

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None], dtype=tf.string, name=tf.saved_model.PREDICT_INPUTS)
    ])
    def predict(serialized_elwcs: tf.Tensor) -> Dict[str, tf.Tensor]:
      """Defines predict signature."""
      features = data.parse_from_example_list(
          serialized_elwcs,
          context_feature_spec=self._context_feature_spec,
          example_feature_spec=self._example_feature_spec,
          mask_feature_name=self._mask_feature_name)
      outputs = self._model(inputs=features, training=False)
      return self.normalize_outputs(tf.saved_model.PREDICT_OUTPUTS, outputs)

    return predict

  def regress_tf_function(self) -> Callable[[tf.Tensor], Dict[str, tf.Tensor]]:
    """Makes a tensorflow function for `regress`."""

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None], dtype=tf.string, name=tf.saved_model.REGRESS_INPUTS)
    ])
    def regress(serialized_examples: tf.Tensor) -> Dict[str, tf.Tensor]:
      """Defines regress signature."""
      features = data.parse_from_tf_example(
          serialized=serialized_examples,
          context_feature_spec=self._context_feature_spec,
          example_feature_spec=self._example_feature_spec,
          mask_feature_name=self._mask_feature_name)
      outputs = tf.nest.map_structure(
          functools.partial(tf.squeeze, axis=1),
          self._model(inputs=features, training=False))
      return self.normalize_outputs(tf.saved_model.REGRESS_OUTPUTS, outputs)

    return regress

  def __call__(
      self,
      serving_default: str = "regress"
  ) -> Dict[str, Callable[[tf.Tensor], Dict[str, tf.Tensor]]]:
    """Returns a dict of signatures.

    Args:
      serving_default: Specifies "regress" or "predict" as the serving_default
        signature.

    Returns:
      A dict of signatures.
    """
    if serving_default not in ["regress", "predict"]:
      raise ValueError("serving_default should be 'regress' or 'predict', "
                       "but got {}".format(serving_default))
    serving_default_function = (
        self.regress_tf_function()
        if serving_default == "regress" else self.predict_tf_function())

    signatures = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            serving_default_function,
        tf.saved_model.REGRESS_METHOD_NAME:
            self.regress_tf_function(),
        tf.saved_model.PREDICT_METHOD_NAME:
            self.predict_tf_function(),
    }
    return signatures
