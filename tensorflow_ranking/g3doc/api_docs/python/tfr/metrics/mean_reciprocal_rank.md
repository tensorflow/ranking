<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.mean_reciprocal_rank" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.mean_reciprocal_rank

Computes mean reciprocal rank (MRR).

```python
tfr.metrics.mean_reciprocal_rank(
    labels,
    predictions,
    weights=None,
    name=None
)
```

Defined in
[`python/metrics.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `predictions`. A value >= 1
    means a relevant example.
*   <b>`predictions`</b>: A `Tensor` with shape [batch_size, list_size]. Each
    value is the ranking score of the corresponding example.
*   <b>`weights`</b>: A `Tensor` of the same shape of predictions or
    [batch_size, 1]. The former case is per-example and the latter case is
    per-list.
*   <b>`name`</b>: A string used as the name for this metric.

#### Returns:

A metric for the weighted mean reciprocal rank of the batch.
