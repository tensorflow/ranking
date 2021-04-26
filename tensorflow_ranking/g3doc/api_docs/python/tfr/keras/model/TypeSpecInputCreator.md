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
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L278-L309">
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
    context_feature_names: Optional[List[str]],
    example_feature_names: Optional[List[str]]
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L291-L309">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__() -> Tuple[TensorDict, TensorDict]
</code></pre>

See `InputCreator`.
