description: Computes average relevance position (ARP).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.average_relevance_position" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.average_relevance_position

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py#L291-L315">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes average relevance position (ARP).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.metrics.average_relevance_position(
    labels, predictions, weights=None, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This can also be named as average_relevance_rank, but this can be confusing with
mean_reciprocal_rank in acronyms. This name is more distinguishing and has been
used historically for binary relevance as average_click_position.

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
A metric for the weighted average relevance position.
</td>
</tr>

</table>
