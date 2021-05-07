description: Computes the percentage of correctly ordered pair.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.ordered_pair_accuracy" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.ordered_pair_accuracy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py#L541-L566">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the percentage of correctly ordered pair.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.metrics.ordered_pair_accuracy(
    labels, predictions, weights=None, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

For any pair of examples, we compare their orders determined by `labels` and
`predictions`. They are correctly ordered if the two orders are compatible. That
is, labels l_i > l_j and predictions s_i > s_j and the weight for this pair is
the weight from the l_i.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` of the same shape as `predictions`.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
A `Tensor` with shape [batch_size, list_size]. Each value is
the ranking score of the corresponding example.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
A `Tensor` of the same shape of predictions or [batch_size, 1]. The
former case is per-example and the latter case is per-list.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string used as the name for this metric.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A metric for the accuracy or ordered pairs.
</td>
</tr>

</table>
