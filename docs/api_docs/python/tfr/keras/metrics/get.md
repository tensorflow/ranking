description: Factory method to get a list of ranking metrics.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.metrics.get" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.metrics.get

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/metrics.py#L53-L111">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Factory method to get a list of ranking metrics.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.metrics.get(
    key: str,
    name: Optional[str] = None,
    dtype: Optional[tf.dtypes.DType] = None,
    topn: Optional[int] = None,
    **kwargs
) -> tf.keras.metrics.Metric
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example Usage:

```python
  metric = tfr.keras.metics.get(tfr.keras.metrics.RankingMetricKey.MRR)
```

to get Mean Reciprocal Rank. `python metric =
tfr.keras.metics.get(tfr.keras.metrics.RankingMetricKey.MRR, topn=2)` to get
MRR@2.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key`
</td>
<td>
An attribute of `RankingMetricKey`, defining which metric objects to
return.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name of metrics.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Dtype of the metrics.
</td>
</tr><tr>
<td>
`topn`
</td>
<td>
Cutoff of how many items are considered in the metric.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments for the metric object.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tf.keras.metrics.Metric. See `_RankingMetric` signature for more details.
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
If key is unsupported.
</td>
</tr>
</table>
