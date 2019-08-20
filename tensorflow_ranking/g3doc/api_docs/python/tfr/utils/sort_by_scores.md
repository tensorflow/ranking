<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.sort_by_scores" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.sort_by_scores

Sorts example features according to per-example scores.

```python
tfr.utils.sort_by_scores(
    scores,
    features_list,
    topn=None
)
```

Defined in
[`python/utils.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`scores`</b>: A `Tensor` of shape [batch_size, list_size] representing
    the per-example scores.
*   <b>`features_list`</b>: A list of `Tensor`s with the same shape as scores to
    be sorted.
*   <b>`topn`</b>: An integer as the cutoff of examples in the sorted list.

#### Returns:

A list of `Tensor`s as the list of sorted features by `scores`.
