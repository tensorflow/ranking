description: Builds a tf.keras.Model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.ModelBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="create_inputs"/>
<meta itemprop="property" content="preprocess"/>
<meta itemprop="property" content="score"/>
</div>

# tfr.keras.model.ModelBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L303-L382">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a `tf.keras.Model`.

Inherits From:
[`ModelBuilderWithMask`](../../../tfr/keras/model/ModelBuilderWithMask.md),
[`AbstractModelBuilder`](../../../tfr/keras/model/AbstractModelBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.ModelBuilder(
    input_creator: Callable[[], Tuple[TensorDict, TensorDict]],
    preprocessor: Callable[[TensorDict, TensorDict, tf.Tensor], Tuple[TensorDict, TensorDict]],
    scorer: Callable[[TensorDict, TensorDict, tf.Tensor], Union[TensorLike, TensorDict]],
    mask_feature_name: str,
    name: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This class implements the `ModelBuilderWithMask` by delegating the class
behaviors to the following implementors that can be specified by callers:

*   input_creator: A callable or a class like `InputCreator` to implement
    `create_inputs`.
*   preprocessor: A callable or a class like `Preprocessor` to implement
    `preprocess`.
*   scorer: A callable or a class like `Scorer` to implement `score`.

Users can subclass those implementor classes and pass the objects into this
class to build a `tf.keras.Model`.

#### Example usage:

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

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_creator`
</td>
<td>
A callable or a class like `InputCreator` to implement
`create_inputs`.
</td>
</tr><tr>
<td>
`preprocessor`
</td>
<td>
A callable or a class like `Preprocessor` to implement
`preprocess`.
</td>
</tr><tr>
<td>
`scorer`
</td>
<td>
A callable or a class like `Scorer` to implement `score`.
</td>
</tr><tr>
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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L359-L364">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_inputs() -> Tuple[TensorDict, TensorDict, tf.Tensor]
</code></pre>

See `ModelBuilderWithMask`.

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L366-L373">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess(
    context_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Tuple[TensorDict, TensorDict]
</code></pre>

See `ModelBuilderWithMask`.

<h3 id="score"><code>score</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L375-L382">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>score(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[TensorLike, TensorDict]
</code></pre>

See `ModelBuilderWithMask`.
