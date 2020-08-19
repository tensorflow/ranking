<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.network.listwise_scoring" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.network.listwise_scoring

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/network.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Listwise scoring op for context and example features.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.network.listwise_scoring(
    scorer, context_features, example_features, training=None, mask=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scorer`
</td>
<td>
A callable (e.g., A keras layer instance, a function) for scoring
with the following signature:
* Args:
`context_features`: (dict) A dict of Tensors with the shape [batch_size,
...].
`example_features`: (dict) A dict of Tensors with the shape [batch_size,
...].
`training`: (bool) whether in training or inference mode.
* Returns: The computed logits, a Tensor of shape [batch_size,
output_size].
</td>
</tr><tr>
<td>
`context_features`
</td>
<td>
(dict) context feature names to dense 2D tensors of shape
[batch_size, ...].
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
(dict) example feature names to dense 3D tensors of shape
[batch_size, list_size, ...].
</td>
</tr><tr>
<td>
`training`
</td>
<td>
(bool) whether in train or inference mode.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
(tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
is True for a valid example and False for invalid one.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
(tf.Tensor) A score tensor of shape [batch_size, list_size, output_size].
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If example features is None or an empty dict.
</td>
</tr>
</table>
