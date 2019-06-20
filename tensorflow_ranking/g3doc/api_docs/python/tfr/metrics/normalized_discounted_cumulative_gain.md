<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.normalized_discounted_cumulative_gain" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.normalized_discounted_cumulative_gain

Computes normalized discounted cumulative gain (NDCG).

```python
tfr.metrics.normalized_discounted_cumulative_gain(
    labels,
    predictions,
    weights=None,
    topn=None,
    name=None
)
```

Defined in
[`python/metrics.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*   <b>`predictions`</b>: A `Tensor` with shape [batch_size, list_size]. Each
    value is the ranking score of the corresponding example.
*   <b>`weights`</b>: A `Tensor` of the same shape of predictions or
    [batch_size, 1]. The former case is per-example and the latter case is
    per-list.
*   <b>`topn`</b>: A cutoff for how many examples to consider for this metric.
*   <b>`name`</b>: A string used as the name for this metric.

#### Returns:

A metric for the weighted normalized discounted cumulative gain of the batch.
