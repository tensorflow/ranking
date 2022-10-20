description: InputCreator with tensor type specs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.TypeSpecInputCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfr.keras.model.TypeSpecInputCreator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L487-L535">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

InputCreator with tensor type specs.

Inherits From: [`InputCreator`](../../../tfr/keras/model/InputCreator.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.TypeSpecInputCreator(
    type_spec: Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]],
    context_feature_names: Optional[List[str]] = None,
    example_feature_names: Optional[List[str]] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example usage:

```python
input_creator=TypeSpecInputCreator(
    {"example_feature_1": tf.TensorSpec(shape=[None, 1], dtype=tf.float32)},
    example_feature_names=["example_feature_1"])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`type_spec`<a id="type_spec"></a>
</td>
<td>
A dict maps the context and example feature keys to the
corresponding context and example type specs.
</td>
</tr><tr>
<td>
`context_feature_names`<a id="context_feature_names"></a>
</td>
<td>
A list of context feature keys.
</td>
</tr><tr>
<td>
`example_feature_names`<a id="example_feature_names"></a>
</td>
<td>
A list of example feature keys.
</td>
</tr>
</table>

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L517-L535">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__() -> Tuple[<a href="../../../tfr/keras/model/TensorDict.md"><code>tfr.keras.model.TensorDict</code></a>, <a href="../../../tfr/keras/model/TensorDict.md"><code>tfr.keras.model.TensorDict</code></a>]
</code></pre>

See `InputCreator`.
