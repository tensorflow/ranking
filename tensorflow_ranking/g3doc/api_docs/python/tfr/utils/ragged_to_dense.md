description: Converts given inputs from ragged tensors to dense tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.ragged_to_dense" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.ragged_to_dense

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py#L402-L424">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Converts given inputs from ragged tensors to dense tensors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.ragged_to_dense(
    labels, predictions, weights
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `tf.RaggedTensor` of the same shape as `predictions` representing
relevance.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
A `tf.RaggedTensor` with shape [batch_size, (list_size)]. Each
value is the ranking score of the corresponding example.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
An optional `tf.RaggedTensor` of the same shape of predictions or a
`tf.Tensor` of shape [batch_size, 1]. The former case is per-example and
the latter case is per-list.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple (labels, predictions, weights, mask) of dense `tf.Tensor`s.
</td>
</tr>

</table>
