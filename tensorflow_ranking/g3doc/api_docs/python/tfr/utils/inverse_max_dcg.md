<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.inverse_max_dcg" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.inverse_max_dcg

Computes the inverse of max DCG.

```python
tfr.utils.inverse_max_dcg(
    labels,
    gain_fn=(lambda labels: tf.pow(2.0, labels) - 1.0),
    rank_discount_fn=(lambda rank: 1.0 / tf.math.log1p(rank)),
    topn=None
)
```

Defined in
[`python/utils.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`labels`</b>: A `Tensor` with shape [batch_size, list_size]. Each value
    is the graded relevance of the corresponding item.
*   <b>`gain_fn`</b>: A gain function. By default this is set to: 2^label - 1.
*   <b>`rank_discount_fn`</b>: A discount function. By default this is set to:
    1/log(1+rank).
*   <b>`topn`</b>: An integer as the cutoff of examples in the sorted list.

#### Returns:

A `Tensor` with shape [batch_size, 1].
