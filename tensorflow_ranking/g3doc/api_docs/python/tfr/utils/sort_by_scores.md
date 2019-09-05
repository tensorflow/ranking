<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.sort_by_scores" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.sort_by_scores

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Sorts example features according to per-example scores.

```python
tfr.utils.sort_by_scores(
    scores,
    features_list,
    topn=None,
    shuffle_ties=True
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`scores`</b>: A `Tensor` of shape [batch_size, list_size] representing
    the per-example scores.
*   <b>`features_list`</b>: A list of `Tensor`s with the same shape as scores to
    be sorted.
*   <b>`topn`</b>: An integer as the cutoff of examples in the sorted list.
*   <b>`shuffle_ties`</b>: A boolean. If True, randomly shuffle before the
    sorting.

#### Returns:

A list of `Tensor`s as the list of sorted features by `scores`.
