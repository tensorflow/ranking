<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.compute_mean" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.compute_mean

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Returns the mean of the specified metric given the inputs.

```python
tfr.metrics.compute_mean(
    metric_key, labels, predictions, weights=None, topn=None, name=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`metric_key`</b>: A key in `RankingMetricKey`.
*   <b>`labels`</b>: A `Tensor` of the same shape as `predictions` representing
    relevance.
*   <b>`predictions`</b>: A `Tensor` with shape [batch_size, list_size]. Each
    value is the ranking score of the corresponding example.
*   <b>`weights`</b>: A `Tensor` of the same shape of predictions or
    [batch_size, 1]. The former case is per-example and the latter case is
    per-list.
*   <b>`topn`</b>: An `integer` specifying the cutoff of how many items are
    considered in the metric.
*   <b>`name`</b>: A `string` used as the name for this metric.

#### Returns:

A scalar as the computed metric.
