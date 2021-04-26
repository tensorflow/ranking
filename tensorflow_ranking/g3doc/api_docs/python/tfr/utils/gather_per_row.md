description: Gathers the values from input tensor based on per-row indices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.gather_per_row" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.gather_per_row

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py#L33-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Gathers the values from input tensor based on per-row indices.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.gather_per_row(
    inputs, indices
)
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example Usage:

```python
scores = [[1., 3., 2.], [1., 2., 3.]]
indices = [[1, 2], [2, 1]]
tfr.utils.gather_per_row(scores, indices)
```
Returns [[3., 2.], [3., 2.]]

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
(tf.Tensor) A tensor of shape [batch_size, list_size] or
[batch_size, list_size, feature_dims].
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
(tf.Tensor) A tensor of shape [batch_size, size] of positions to
gather inputs from. Each index corresponds to a row entry in input_tensor.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of values gathered from inputs, of shape [batch_size, size] or
[batch_size, size, feature_dims], depending on whether the input was 2D or
3D.
</td>
</tr>

</table>
