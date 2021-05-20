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
"""Defines Keras Layers for TF-Ranking."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import tensorflow as tf

from official.nlp.modeling import layers as nlp_modeling_layers
from tensorflow_ranking.python import utils

_EPSILON = 1e-10


def create_tower(hidden_layer_dims: List[int],
                 output_units: int,
                 activation: Optional[Callable[..., tf.Tensor]] = None,
                 input_batch_norm: bool = False,
                 use_batch_norm: bool = True,
                 batch_norm_moment: float = 0.999,
                 dropout: float = 0.5,
                 name: Optional[str] = None,
                 **kwargs: Dict[Any, Any]):
  """Creates a feed-forward network as `tf.keras.Sequential`.

  It creates a feed-forward network with batch normalization and dropout, and
  optionally applies batch normalization on inputs.

  Example usage:
  ```python
  tower = create_tower(hidden_layer_dims=[64, 32, 16], output_units=1)
  inputs = tf.ones([2, 3, 1])
  tower_logits = tower(inputs)
  ```

  Args:
    hidden_layer_dims: Iterable of number hidden units per layer. All layers are
      fully connected. Ex. `[64, 32]` means first layer has 64 nodes and second
      one has 32.
    output_units: Size of output logits from this tower.
    activation: Activation function applied to each layer. If `None`, will use
      an identity activation.
    input_batch_norm: Whether to use batch normalization for input layer.
    use_batch_norm: Whether to use batch normalization after each hidden layer.
    batch_norm_moment: Momentum for the moving average in batch normalization.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    name: Name of the Keras layer.
    **kwargs: Keyword arguments for every `tf.keras.Dense` layers.

  Returns:
    A `tf.keras.Sequential` object.
  """
  model = tf.keras.Sequential(name=name)
  # Input batch normalization.
  if input_batch_norm:
    model.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_moment))
  for layer_width in hidden_layer_dims:
    model.add(tf.keras.layers.Dense(units=layer_width), **kwargs)
    if use_batch_norm:
      model.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_moment))
    model.add(tf.keras.layers.Activation(activation=activation))
    if dropout:
      model.add(tf.keras.layers.Dropout(rate=dropout))
  model.add(tf.keras.layers.Dense(units=output_units), **kwargs)
  return model


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class FlattenList(tf.keras.layers.Layer):
  """Layer to flatten the example list.

  This layer flattens the batch_size dimension and the list_size dimension for
  the `example_features` and expands list_size times for the `context_features`.

  Example usage:

  ```python
  context_features = {'context_feature_1': [[1], [0]]}
  example_features = {'example_feature_1':
                      [[[1], [0], [-1]], [[0], [1], [0]]]}
  mask = [[True, True, False], [True, False, False]]
  flattened_context_features, flattened_example_features = FlattenList()(
      inputs=(context_features, example_features, mask))
  ```
  That is, there are two valid examples in the first query and one
  valid example in the second query. Then

  ```python
  flattened_context_features = {'context_feature_1':
                                [[1], [1], [1], [0], [0], [0]]}
  flattened_example_features = {'example_feature_1':
                                [[1], [0], [1], [0], [0], [0]]}
  ```
  `context_feature_1` is repeated by list_size=3 times. `example_feature_1` is
  flattened and padded with the invalid terms replaced by valid terms in each
  query in a circular way.
  """

  def __init__(self,
               circular_padding: bool = True,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the FlattenList layer.

    Args:
      circular_padding: Whether to apply circular padding to replace invalid
        features with valid ones.
      name: Name of the layer.
      **kwargs: keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._circular_padding = circular_padding

  def call(
      self, inputs: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Call FlattenList layer to flatten context_features and example_features.

    Args:
      inputs: A tuple of (context_features, example_features, list_mask), which
        are described below:
      * `context_features`: A map of context features to 2D tensors of shape
        [batch_size, feature_dim].
      * `example_features`: A map of example features to 3D tensors of shape
        [batch_size, list_size, feature_dim].
      * `list_mask`: A Tensor of shape [batch_size, list_size] to mask out the
        invalid examples.

    Returns:
      A tuple of (flattened_context_features, flattened_example_fatures) where
      the former is a dict of context features to 2D tensors of shape
      [batch_size * list_size, feature_dim] and the latter is a dict of example
      features to 2D tensors of shape [batch_size * list_size, feature_dim].

    Raises:
      ValueError: If `example_features` is empty dict or None.
    """
    context_features, example_features, list_mask = inputs
    if not example_features:
      raise ValueError('Need a valid example feature.')
    batch_size = tf.shape(list_mask)[0]
    list_size = tf.shape(list_mask)[1]
    # Expand context features to be of [batch_size, list_size, ...].
    flattened_context_features = {}
    for name, tensor in context_features.items():
      expanded_tensor = tf.repeat(
          tf.expand_dims(tensor, axis=1), repeats=[list_size], axis=1)
      flattened_context_features[name] = utils.reshape_first_ndims(
          expanded_tensor, 2, [batch_size * list_size])

    nd_indices = None
    if self._circular_padding:
      nd_indices, _ = utils.padded_nd_indices(is_valid=list_mask)

    flattened_example_features = {}
    for name, tensor in example_features.items():
      if nd_indices is not None:
        # Replace invalid example features with valid ones.
        tensor = tf.gather_nd(tensor, nd_indices)
      flattened_example_features[name] = utils.reshape_first_ndims(
          tensor, 2, [batch_size * list_size])

    return flattened_context_features, flattened_example_features

  def get_config(self):
    config = super().get_config()
    config.update({
        'circular_padding': self._circular_padding,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class RestoreList(tf.keras.layers.Layer):
  """Output layer to restore listwise output shape.

  This layer maps the flattened 1D logits or 2D tensor of shape
  [batch_size * list_size, 1] back to 2D of shape [batch_size, list_size] and
  mask the invalid terms to be a defined large negative value.

  Example usage:

  ```python
  flattened_logits = [1, 0.5, 2, 0, -1, 0]
  mask = [[True, True, False], [True, False, False]]
  logits = RestoreList()(inputs=(flattened_logits, mask))
  logits.numpy()
  # Returns: [[1, 0.5, log(1e-10)], [0, log(1e-10), log(1e-10)]]
  ```

  This layer works also for 2D `flattened_logits` like
  [[1], [0.5], [2], [0], [-1], [0]].

  When `by_scatter=True`, an nd_indices will be generated using `mask` in the
  same way as `FlattenList`. All values in the `flattened_logits` will be used
  and repeated entries will be averaged.

  ```python
  flattened_logits = [1, 0.5, 2, 0, -1, 0]
  mask = [[True, True, False], [True, False, False]]
  logits = RestoreList(by_scatter=True)((flattened_logits, mask))
  logits.numpy()
  # Returns: [[1.5, 0.5, log(1e-10)], [-1/3, log(1e-10), log(1e-10)]]
  ```

  This is because the flattened_logits are treated as circularly padded entries.
  The [1st, 3rd] values [1, 2] are counted to logits[0, 0]. The [4th, 5th, 6th]
  values [0, -1, 0] are counted to logits[1, 0]. Note that We use different
  values for those repeated entries, while they are likely the same in practice.
  """

  def __init__(self,
               name: Optional[str] = None,
               by_scatter: bool = False,
               **kwargs: Dict[Any, Any]):
    super().__init__(name=name, **kwargs)
    self._by_scatter = by_scatter

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Restores listwise shape of flattened_logits.

    Args:
      inputs:  A tuple of (flattened_logits, list_mask), which are described
        below.
      * `flattened_logits`: A `Tensor` of predicted logits for each pair of
        query and documents, 1D tensor of shape [batch_size * list_size] or 2D
        tensor of shape [batch_size * list_size, 1].
      * `list_mask`: A boolean `Tensor` of shape [batch_size, list_size] to mask
        out the invalid examples.

    Returns:
      A `Tensor` of shape [batch_size, list_size].

    Raises:
      ValueError: If `flattened_logits` is not of shape [batch_size * list_size]
        or [batch_size * list_size, 1].
    """
    flattened_logits, list_mask = inputs
    try:
      logits = tf.reshape(flattened_logits, shape=tf.shape(list_mask))
    except:
      raise ValueError('`flattened_logits` needs to be either '
                       '1D of [batch_size * list_size] or '
                       '2D of [batch_size * list_size, 1].')
    if self._by_scatter:
      nd_indices, _ = utils.padded_nd_indices(is_valid=list_mask)
      counts = tf.scatter_nd(nd_indices, tf.ones_like(logits),
                             tf.shape(list_mask))
      logits = tf.scatter_nd(nd_indices, logits, tf.shape(list_mask))
      return tf.where(
          tf.math.greater(counts, 0.), logits / counts, tf.math.log(_EPSILON))
    else:
      return tf.where(list_mask, logits, tf.math.log(_EPSILON))

  def get_config(self):
    config = super().get_config()
    config.update({
        'by_scatter': self._by_scatter,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ConcatFeatures(tf.keras.layers.Layer):
  """Concatenates context features and example features in a listwise manner.

  Given dicts of `context features`, `example features`, this layer expands
  list_size times for the `context_features` and concatenates them with
  example features along the `list_size` axis. The output is a 3-d tensor with
  shape [batch_size, list_size, sum(feature_dims)], where sum(feature_dims) is
  the sum of all example feature dimensions and the context feature dimension.

  Example usage:

  ConcatFeatures with circular padding.

  ```python
  # Batch size = 2, list_size = 2.
  context_features = {
      'context_feature_1': [[1.], [2.]]
  }
  example_features = {
        'example_feature_1':
            [[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]
  }
  mask = [[True, False], [True, True]]
  concat_tensor = ConcatFeatures()(inputs=(context_features,
      example_features, mask))
  concat_tensor.numpy()
  # Returns: [[[1., 1., 0.], [1., 1., 0.]], [[2., 0., 1.], [2., 1., 0.]]])
  ```

  ConcatFeatures without circular padding.

  ```python
  concat_tensor = ConcatFeatures(circular_padding=False)(
     inputs=(context_features, example_features, mask))
  concat_tensor.numpy()
  # Returns: [[[1., 1., 0.], [1., 0., 1.]], [[2., 0., 1.], [2., 1., 0.]]]
  ```
  """

  def __init__(self,
               circular_padding: bool = True,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the ConcatFeatures layer.

    Args:
      circular_padding: Whether to apply circular padding to replace invalid
        features with valid ones.
      name: Name of the layer.
      **kwargs: keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._circular_padding = circular_padding
    self._flatten_list = FlattenList(circular_padding=self._circular_padding)

  def call(
      self, inputs: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Call layer to flatten context_features and example_features.

    Args:
      inputs: A tuple of (context_features, example_features, list_mask), which
        are described below:
      * `context_features`: A map of context features to 2D tensors of shape
        [batch_size, feature_dim].
      * `example_features`: A map of example features to 3D tensors of shape
        [batch_size, list_size, feature_dim].
      * `list_mask`: A Tensor of shape [batch_size, list_size] to mask out the
        invalid examples.

    Returns:
      A tensor of shape [batch_size, list_size, concat_feature_dim].
    """
    context_features, example_features, list_mask = inputs
    (flattened_context_features,
     flattened_example_features) = self._flatten_list(
         (context_features, example_features, list_mask))
    # Concatenate flattened context and example features along `list_size` dim.
    context_input = [
        tf.keras.layers.Flatten()(flattened_context_features[name])
        for name in sorted(flattened_context_features)
    ]
    example_input = [
        tf.keras.layers.Flatten()(flattened_example_features[name])
        for name in sorted(flattened_example_features)
    ]
    flattened_concat_features = tf.concat(context_input + example_input, 1)

    # Reshape to 3D.
    batch_size = tf.shape(list_mask)[0]
    list_size = tf.shape(list_mask)[1]
    return utils.reshape_first_ndims(flattened_concat_features, 1,
                                     [batch_size, list_size])

  def get_config(self):
    config = super().get_config()
    config.update({
        'circular_padding': self._circular_padding,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class DocumentInteractionAttention(tf.keras.layers.Layer):
  """Cross Document Interaction Attention layer.

  This layer implements the cross-document attention described in
  [Pasumarthi et al, 2020][pasumarthi2020].

  This layer comprises of several layers of Multi-Headed Attention (MHA)
  applied over the list of documents to attend over itself, using a mask to
  specify valid documents. Optionally, the user can specify the `topk` documents
  as the listwise context that is used to compute the attention per document. If
  `topk` is None, all the documents are used as listwise context to compute the
  attention.

  MHA uses scaled dot product attention, with residual connection and layer
  normalization as follows. This transformation is applied for `num_layers`
  times:

  ```
  h_i := LayerNorm_i(h_{i-1} + MHA_i(h_{i-1}), TopK(h_{i-1}; k))
  ```

  Example usage:

  ```python
  # Batch size = 2, list_size = 3.
  inputs =  [[[1., 1.], [1., 0.], [1., 1.]], [[0., 0.], [0., 0.], [0., 0.]]]
  list_mask = [[True, True, False], [True, False, False]]
  dia_layer = DocumentInteractionAttention(
      num_heads=1, head_size=64, num_layers=1)
  dia_output = dia_layer(
      inputs=inputs,
      training=False,
      list_mask=list_mask)
  ```

  References:
    - [Permutation Equivariant Document Interaction Network for Neural Learning
       to Rank, Pasumarthi et al, 2020][pasumarthi2020]

  [pasumarthi2020]: http://research.google/pubs/pub49364/
  """

  def __init__(self,
               num_heads: int,
               head_size: int,
               num_layers: int = 1,
               dropout: float = 0.5,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the layer.

    Args:
      num_heads: Number of attention heads (see `MultiHeadAttention` for more
        details on this argument).
      head_size: Size of each attention head.
      num_layers: Number of cross-document attention layers.
      dropout: Dropout probability.
      name: Name of the layer.
      **kwargs: keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._num_heads = num_heads
    self._head_size = head_size
    self._num_layers = num_layers
    self._dropout = dropout

  def build(self, input_shape: tf.TensorShape):
    """Build method to create weights and sub-layers.

    This method can be used to create weights that depend on the shape of the
    input(s), using add_weight().
    `__call__()` will automatically build the layer by calling `build()`.

    Args:
      input_shape: A tuple of shapes for `example_inputs`, `list_mask`. These
      correspond to `inputs` argument of call.
    """
    example_inputs_shape, list_mask_shape = input_shape
    example_inputs_shape = tf.TensorShape(example_inputs_shape)
    list_mask_shape = tf.TensorShape(list_mask_shape)
    din_embedding_shape = tf.TensorShape(
        [example_inputs_shape[0], example_inputs_shape[1], self._head_size])

    # This projects input to head_size, so that this layer can be applied
    # recursively for `num_layers` times.
    # Shape: [batch_size, list_size, feature_dims] ->
    # [batch_size, list_size, head_size].
    self._input_projection = tf.keras.layers.Dense(
        units=self._head_size, activation='relu')
    self._input_projection.build(example_inputs_shape)

    # Self-attention layers.
    self._attention_layers = []
    for _ in range(self._num_layers):
      # Shape: [batch_size, list_size, head_size] ->
      # [batch_size, list_size, head_size].
      attention_layer = tf.keras.layers.MultiHeadAttention(
          num_heads=self._num_heads,
          key_dim=self._head_size,
          dropout=self._dropout,
          output_shape=self._head_size)

      # pylint: disable=protected-access
      attention_layer._build_from_signature(
          query=din_embedding_shape, value=din_embedding_shape)
      # pylint: enable=protected-access

      # Dropout and layer normalization are applied element-wise, and do not
      # change the shape.
      dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)
      norm_layer = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, dtype=tf.float32)
      self._attention_layers.append(
          (attention_layer, dropout_layer, norm_layer))
    super().build(input_shape)

  def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
    """Calls the document interaction layer to apply cross-document attention.

    Args:
      inputs: A tuple of `example_inputs`, `list_mask`.
        `example_inputs`: A tensor of shape [batch_size, list_size,
          feature_dims].
        `list_mask`: A boolean tensor of shape [batch_size, list_size], which is
        True for a valid example and False for invalid one. If this is `None`,
        then all examples are treated as valid.
      training: Whether in training or inference mode.

    Returns:
      A tensor of shape [batch_size, list_size, head_size].
    """
    example_inputs, list_mask = inputs
    batch_size = tf.shape(example_inputs)[0]
    list_size = tf.shape(example_inputs)[1]
    if list_mask is None:
      list_mask = tf.ones(shape=(batch_size, list_size), dtype=tf.bool)
    x = self._input_projection(example_inputs, training=training)

    list_mask = tf.cast(list_mask, dtype=tf.int32)
    attention_mask = nlp_modeling_layers.SelfAttentionMask()(
        [list_mask, list_mask])

    for attention_layer, dropout_layer, norm_layer in self._attention_layers:
      output = attention_layer(
          query=x, value=x, attention_mask=attention_mask, training=training)
      output = dropout_layer(output, training=training)
      # Applying residual network here, similar to logic in Transformer.
      x = norm_layer(output + x, training=training)

    return x

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_heads': self._num_heads,
        'head_size': self._head_size,
        'num_layers': self._num_layers,
        'dropout': self._dropout,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class GAMLayer(tf.keras.layers.Layer):
  r"""Defines a generalized additive model (GAM) layer.

  This layer implements the neural generalized additive ranking model described
  in [Zhuang et al, 2021][zhuang2021].

  Neural Generalized Additive Ranking Model is an additive ranking model.
  For each example `x` with `n` features `(x_1, x_2, ..., x_n)`, the ranking
  score is:

    $$F(x) = f_1(x_1) + f_2(x_2) + \ldots + f_n(x_n)$$

  where each feature is scored by a corresponding submodel, and the overall
  ranking score is the sum of all the submodels' outputs. Each submodel is a
  standalone feed-forward network.

  When there are `m` context features `(c_1, c_2, ..., c_m)`, the ranking score
  will be determined by:

    $$F(c, x) = w_1(c) * f_1(x_1) + w_2(c) * f_2(x_2) + \ldots + w_n(c) *
    f_n(x_n)$$

  where `(w1(c), w2(c), ..., wn(c))` is a weighting vector determined solely by
  context features. For each context feature `c_j`, a feed-forward submodel is
  constructed to derive a weighting vector
  `(wj1(c_j), wj2(c_j), ..., wjn(c_j))`. The final weighting vector is the sum
  of the output of all the context features' submodels.

  The model is implicitly interpretable as the contribution of each feature to
  the final ranking score can be easily visualized. However, the model does not
  have higher-order inter-feature interactions and hence may not have
  performance as good as a fully-connected DNN.

  The output of each example feature's submodel can be retrieved by tensor
  named `{feature_name}_subscore`. The output of each context feature's submodel
  is a n-dimensional vector and can be retrieved by tensor named
  `{feature_name}_subweight`.

  ```python
  example_inputs = tf.constant([[1], [0], [-1]], dtype=tf.float32)
  context_inputs = tf.constant([[1, 2], [0, 1], [-1, 1]], dtype=tf.float32)
  gam = layers.GAMLayer(2, [3, 2, 1], 2, [3, 2, 1])
  outputs, sublogits_list, subweights_list = gam(
      ([example_inputs, example_inputs], [context_inputs, context_inputs]))
  ```

  References:
    - [Interpretable Ranking with Generalized Additive Models, Zhuang et al,
       2021][zhuang2021]

  [zhuang2021]: https://research.google/pubs/pub50040/
  """

  def __init__(self,
               example_feature_num: int,
               example_hidden_layer_dims: List[int],
               context_feature_num: Optional[int] = None,
               context_hidden_layer_dims: Optional[List[int]] = None,
               activation: Optional[Callable[..., tf.Tensor]] = None,
               use_batch_norm: bool = True,
               batch_norm_moment: float = 0.999,
               dropout: float = 0.5,
               name: Optional[str] = None,
               **kwargs: Dict[Any, Any]):
    """Initializes the layer.

    Args:
      example_feature_num: Number of example features.
      example_hidden_layer_dims: Iterable of number hidden units for an tower.
        Each example feature will have an identical tower.
      context_feature_num: Number of context features. If `None` or 0 then no
        context weighting will be applied, otherwise `context_hidden_layer_dims`
        is required.
      context_hidden_layer_dims: Iterable of number hidden units for an tower.
        Each context feature (if any) will have an identical tower. Required if
        `context_feature_num` is specified.
      activation: Activation function applied to each layer. If `None`, will use
        an identity activation.
      use_batch_norm: Whether to use batch normalization after each hidden
        layer.
      batch_norm_moment: Momentum for the moving average in batch normalization.
      dropout: When not `None`, the probability of dropout for the dropoout
        layer in each tower.
      name: Name of the Keras layer.
      **kwargs: Keyword arguments.
    """

    super().__init__(name=name, **kwargs)
    self._example_feature_num = example_feature_num
    self._context_feature_num = context_feature_num
    self._example_hidden_layer_dims = example_hidden_layer_dims
    self._context_hidden_layer_dims = context_hidden_layer_dims
    self._activation = tf.keras.activations.get(activation)
    self._use_batch_norm = use_batch_norm
    self._batch_norm_moment = batch_norm_moment
    self._dropout = dropout

    self._example_towers = []
    for i in range(self._example_feature_num):
      self._example_towers.append(
          create_tower(
              hidden_layer_dims=self._example_hidden_layer_dims,
              output_units=1,
              activation=self._activation,
              use_batch_norm=self._use_batch_norm,
              batch_norm_moment=self._batch_norm_moment,
              dropout=self._dropout,
              name='{}_example_tower_{}'.format(name, i)))

    self._context_towers = None
    if context_feature_num and context_feature_num > 0:
      if not context_hidden_layer_dims:
        raise ValueError(
            'When `context_feature_num` > 0, `context_hidden_layer_dims` is '
            'required! Currently `context_feature_num` is {}, but '
            '`context_hidden_layer_dims` is {}'.format(
                context_feature_num, context_hidden_layer_dims))
      self._context_towers = []
      for i in range(self._context_feature_num):
        self._context_towers.append(
            create_tower(
                hidden_layer_dims=self._context_hidden_layer_dims,
                output_units=self._example_feature_num,
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                batch_norm_moment=self._batch_norm_moment,
                dropout=self._dropout,
                name='{}_context_tower_{}'.format(name, i)))

  def call(
      self,
      inputs: Tuple[List[tf.Tensor], Optional[List[tf.Tensor]]],
      training: bool = True
  ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
    """Obtains the outputs of the GAM model.

    Args:
      inputs: A tuple of (`example_inputs`, `context_inputs`):
      * `example_inputs`: An iterable of Tensors where each tensor is 2-D with
        the shape [batch_size, ...]. The number of tensors should align with
        `example_feature_num`.
      * `context_inputs`: An iterable of Tensors where each tensor is 2-D with
        the shape [batch_size, ...]. If given, the number of tensors should
        align with `context_feature_num`. Notice that even if
        `context_feature_num` is larger than zero, one can still call without
        `context_inputs`. In this case the sub_logits from examples features
        will be directly added and context feature towers will be ignored.
      training: Whether training or not.

    Returns:
      The final scores from the GAM model, lists of tensors representing the
      sublogits of each example feature, and lists of tensors representing the
      subweights derived from each context feature. If no `context_inputs` are
      given, the third element will be an empty list.

    Raises:
      ValueError: An error occurred when the number of tensors in
        `example_inputs` is different from `example_feature_num`.
      ValueError: An error occurred when `context_inputs` is given but the
        number of tensors in `context_inputs` is different from
        `context_feature_num`.
    """
    example_inputs, context_inputs = inputs
    if len(example_inputs) != self._example_feature_num:
      raise ValueError('Mismatched number of features in `example_inputs` ({}) '
                       'with `example_feature_num` ({})'.format(
                           len(example_inputs), self._example_feature_num))
    if context_inputs:
      if (not self._context_towers or
          len(context_inputs) != len(self._context_towers)):
        raise ValueError('Mismatched number of features in `context_inputs` '
                         '({}) with `_context_feature_num` ({})'.format(
                             len(context_inputs), self._context_feature_num))

    sub_logits_list = []
    for inputs, tower in zip(example_inputs, self._example_towers):
      sub_logits = tower(inputs, training=training)
      sub_logits_list.append(sub_logits)

    sub_weights_list = []
    if context_inputs and self._context_towers:
      for inputs, tower in zip(context_inputs, self._context_towers):
        cur = tower(inputs, training=training)
        sub_weights = tf.keras.layers.Softmax()(cur)
        sub_weights_list.append(sub_weights)

    # Construct an additive model from the outputs of all example feature towers
    # weighted by outputs of all context feature towers.
    if sub_weights_list:
      sub_logits = tf.keras.layers.Concatenate(axis=-1)(sub_logits_list)
      sub_weights = (
          tf.keras.layers.Add()(sub_weights_list)
          if len(sub_weights_list) > 1 else sub_weights_list[0])
      logits = tf.reduce_sum(sub_logits * sub_weights, axis=-1, keepdims=True)
    else:
      logits = tf.keras.layers.Add()(
          sub_logits_list) if len(sub_logits_list) > 1 else sub_logits_list[0]

    return logits, sub_logits_list, sub_weights_list

  def get_config(self):
    config = super().get_config()
    config.update({
        'example_feature_num': self._example_feature_num,
        'context_feature_num': self._context_feature_num,
        'example_hidden_layer_dims': self._example_hidden_layer_dims,
        'context_hidden_layer_dims': self._context_hidden_layer_dims,
        'activation': tf.keras.activations.serialize(self._activation),
        'use_batch_norm': self._use_batch_norm,
        'batch_norm_moment': self._batch_norm_moment,
        'dropout': self._dropout
    })
    return config
