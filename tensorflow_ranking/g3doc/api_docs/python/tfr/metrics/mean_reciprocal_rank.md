<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.mean_reciprocal_rank" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.mean_reciprocal_rank

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

<!-- Start diff -->

Computes mean reciprocal rank (MRR).

```python
tfr.metrics.mean_reciprocal_rank(
    labels,
    predictions,
    weights=None,
    topn=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `predictions`. A value >= 1
    means a relevant example.
*   <b>`predictions`</b>: A `Tensor` with shape [batch_size, list_size]. Each
    value is the ranking score of the corresponding example.
*   <b>`weights`</b>: A `Tensor` of the same shape of predictions or
    [batch_size, 1]. The former case is per-example and the latter case is
    per-list.
*   <b>`topn`</b>: An integer cutoff specifying how many examples to consider
    for this metric. If None, the whole list is considered.
*   <b>`name`</b>: A string used as the name for this metric.

#### Returns:

A metric for the weighted mean reciprocal rank of the batch.
