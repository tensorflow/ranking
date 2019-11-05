<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.make_ranking_metric_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.make_ranking_metric_fn

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

<!-- Start diff -->

Factory method to create a ranking metric function.

```python
tfr.metrics.make_ranking_metric_fn(
    metric_key,
    weights_feature_name=None,
    topn=None,
    name=None,
    gain_fn=_DEFAULT_GAIN_FN,
    rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`metric_key`</b>: A key in `RankingMetricKey`.
*   <b>`weights_feature_name`</b>: A `string` specifying the name of the weights
    feature in `features` dict.
*   <b>`topn`</b>: An `integer` specifying the cutoff of how many items are
    considered in the metric.
*   <b>`name`</b>: A `string` used as the name for this metric.
*   <b>`gain_fn`</b>: (function) Transforms labels. A method to calculate gain
    parameters used in the definitions of the DCG and NDCG metrics, where the
    input is the relevance label of the item. The gain is often defined to be of
    the form 2^label-1.
*   <b>`rank_discount_fn`</b>: (function) The rank discount function. A method
    to define the dicount parameters used in the definitions of DCG and NDCG
    metrics, where the input in the rank of item. The discount function is
    commonly defined to be of the form log(rank+1).

#### Returns:

A metric fn with the following Args: * `labels`: A `Tensor` of the same shape as
`predictions` representing graded relevance. * `predictions`: A `Tensor` with
shape [batch_size, list_size]. Each value is the ranking score of the
corresponding example. * `features`: A dict of `Tensor`s that contains all
features.
