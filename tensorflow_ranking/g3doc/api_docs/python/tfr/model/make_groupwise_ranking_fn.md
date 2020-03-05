<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.model.make_groupwise_ranking_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.model.make_groupwise_ranking_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/model.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Builds an `Estimator` model_fn for groupwise comparison ranking models.

```python
tfr.model.make_groupwise_ranking_fn(
    group_score_fn, group_size, ranking_head, transform_fn=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`group_score_fn`</b>: See `_GroupwiseRankingModel`.
*   <b>`group_size`</b>: See `_GroupwiseRankingModel`.
*   <b>`ranking_head`</b>: A `head._RankingHead` object.
*   <b>`transform_fn`</b>: See `_GroupwiseRankingModel`.

#### Returns:

See `_make_model_fn`.
