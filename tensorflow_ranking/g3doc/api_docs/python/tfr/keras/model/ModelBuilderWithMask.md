description: Interface to build a tf.keras.Model for ranking with a mask Tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.ModelBuilderWithMask" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="create_inputs"/>
<meta itemprop="property" content="preprocess"/>
<meta itemprop="property" content="score"/>
</div>

# tfr.keras.model.ModelBuilderWithMask

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L100-L300">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface to build a `tf.keras.Model` for ranking with a mask Tensor.

Inherits From:
[`AbstractModelBuilder`](../../../tfr/keras/model/AbstractModelBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.ModelBuilderWithMask(
    mask_feature_name: str,
    name: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The `ModelBuilderWithMask` class is an abstract class to build a ranking model
based on dense Tensors and a mask Tensor to indicate the padded ones. All the
boilerplate codes related to constructing a `tf.keras.Model` are integrated in
the ModelBuilder class.

To be implemented by subclasses:

*   `create_inputs()`: Contains the logic to create `tf.keras.Input` for context
    and example inputs and mask for valid list items.
*   `preprocess()`: Contains the logic to preprocess context and example inputs.
*   `score()`: Contains the logic to score examples in list and return outputs.

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

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mask_feature_name`
</td>
<td>
name of 2D mask boolean feature.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(optional) name of the Model.
</td>
</tr>
</table>

## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L272-L300">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build() -> tf.keras.Model
</code></pre>

Builds a Keras Model for Ranking Pipeline.

#### Example usage:

```python
model_builder = SimpleModelBuilder(
    {},
    {"example_feature_1": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=0.0)},
    "list_mask", "model_builder")
model = model_builder.build()
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.keras.Model`.
</td>
</tr>

</table>

<h3 id="create_inputs"><code>create_inputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L177-L198">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>create_inputs() -> Tuple[TensorDict, TensorDict, tf.Tensor]
</code></pre>

Creates context and example inputs.

#### Example usage:

```python
model_builder = SimpleModelBuilder(
    {},
    {"example_feature_1": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=0.0)},
    "list_mask", "model_builder")
context_inputs, example_inputs, mask = model_builder.create_inputs()
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of
</td>
</tr>
<tr>
<td>
`context_inputs`
</td>
<td>
maps from context feature keys to Keras Input.
</td>
</tr><tr>
<td>
`example_inputs`
</td>
<td>
maps from example feature keys to Keras Input.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
Keras Input for the mask feature.
</td>
</tr>
</table>

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L200-L234">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>preprocess(
    context_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Tuple[TensorDict, TensorDict]
</code></pre>

Preprocesses context and example inputs.

#### Example usage:

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

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_inputs`
</td>
<td>
maps context feature keys to `tf.keras.Input`.
</td>
</tr><tr>
<td>
`example_inputs`
</td>
<td>
maps example feature keys to `tf.keras.Input`.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
[batch_size, list_size]-tensor of mask for valid examples.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of
</td>
</tr>
<tr>
<td>
`context_features`
</td>
<td>
maps from context feature keys to [batch_size,
feature_dims]-tensors of preprocessed context features.
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
maps from example feature keys to [batch_size,
list_size, feature_dims]-tensors of preprocessed example features.
</td>
</tr>
</table>

<h3 id="score"><code>score</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L236-L270">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>score(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[TensorLike, TensorDict]
</code></pre>

Scores all examples and returns outputs.

#### Example usage:

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

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_features`
</td>
<td>
maps from context feature keys to [batch_size,
feature_dims]-tensors of preprocessed context features.
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
maps from example feature keys to [batch_size,
list_size, feature_dims]-tensors of preprocessed example features.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
[batch_size, list_size]-tensor of mask for valid examples.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A [batch_size, list_size]-tensor of logits or a dict mapping task name to
logits in the multi-task setting.
</td>
</tr>

</table>
