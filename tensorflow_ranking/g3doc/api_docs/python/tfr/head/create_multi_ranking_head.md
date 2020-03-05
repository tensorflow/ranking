<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.head.create_multi_ranking_head" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.head.create_multi_ranking_head

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/head.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

A factory method to create `_MultiRankingHead`.

```python
tfr.head.create_multi_ranking_head(
    heads, head_weights=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`heads`</b>: A tuple or list of `_RankingHead`.
*   <b>`head_weights`</b>: A tuple or list of weights.

#### Returns:

An instance of `_MultiRankingHead` for multi-task learning.
