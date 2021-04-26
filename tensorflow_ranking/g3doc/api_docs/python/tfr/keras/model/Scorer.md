description: Interface for scorer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.Scorer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
</div>

# tfr.keras.model.Scorer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L388-L411">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for scorer.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L391-L411">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__call__(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[TensorLike, TensorDict]
</code></pre>

Scores all examples given context and returns logits.

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
