description: Computes binary preference (BPref).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.binary_preference" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.binary_preference

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py#L569-L605">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes binary preference (BPref).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.metrics.binary_preference(
    labels, predictions, weights=None, topn=None, name=None, use_trec_version=True
)
</code></pre>

<!-- Placeholder for "Used in" -->

The implementation of BPref is based on the desciption in the following:
https://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf BPref = 1 / R
SUM_r(1 - |n ranked higher than r| / min(R, N))

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` of the same shape as `predictions`. A value >= 1 means a
relevant example.
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
`topn`
</td>
<td>
A cutoff for how many examples to consider for this metric.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string used as the name for this metric.
</td>
</tr><tr>
<td>
`use_trec_version`
</td>
<td>
A boolean to choose the version of the formula to use.
If False, than the alternative BPref formula will be used:
BPref = 1 / R SUM_r(1 - |n ranked higher than r| / R)
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A metric for binary preference metric of the batch.
</td>
</tr>

</table>
