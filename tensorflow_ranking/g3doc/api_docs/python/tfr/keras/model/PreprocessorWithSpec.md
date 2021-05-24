description: Preprocessing inputs with provided spec.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.PreprocessorWithSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfr.keras.model.PreprocessorWithSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L588-L652">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Preprocessing inputs with provided spec.

Inherits From: [`Preprocessor`](../../../tfr/keras/model/Preprocessor.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.PreprocessorWithSpec(
    preprocess_spec: Optional[Dict[str, Callable[[Any], Any]]] = None,
    default_value_spec: Optional[Dict[str, float]] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

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

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`preprocess_spec`
</td>
<td>
maps a feature key to a callable to preprocess a feature.
Only include those features that need preprocessing.
</td>
</tr><tr>
<td>
`default_value_spec`
</td>
<td>
maps a feature key to a default value to convert a
RaggedTensor to Tensor. Default to 0. if not specified.
</td>
</tr>
</table>

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L622-L652">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    context_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Tuple[TensorDict, TensorDict]
</code></pre>

See `Preprocessor`.
