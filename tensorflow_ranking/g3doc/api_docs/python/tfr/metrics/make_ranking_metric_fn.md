<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.make_ranking_metric_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.make_ranking_metric_fn

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Factory method to create a ranking metric function.

```python
tfr.metrics.make_ranking_metric_fn(
    metric_key,
    weights_feature_name=None,
    topn=None,
    name=None
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

#### Returns:

A metric fn with the following Args: * `labels`: A `Tensor` of the same shape as
`predictions` representing graded relevance. * `predictions`: A `Tensor` with
shape [batch_size, list_size]. Each value is the ranking score of the
corresponding example. * `features`: A dict of `Tensor`s that contains all
features.
