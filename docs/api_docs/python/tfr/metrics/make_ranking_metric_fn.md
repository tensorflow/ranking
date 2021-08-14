description: Factory method to create a ranking metric function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.make_ranking_metric_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.make_ranking_metric_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py#L102-L259">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Factory method to create a ranking metric function.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.metrics.make_ranking_metric_fn(
    metric_key, weights_feature_name=None, topn=None, name=None,
    gain_fn=_DEFAULT_GAIN_FN, rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`metric_key`
</td>
<td>
A key in `RankingMetricKey`.
</td>
</tr><tr>
<td>
`weights_feature_name`
</td>
<td>
A `string` specifying the name of the weights feature
in `features` dict.
</td>
</tr><tr>
<td>
`topn`
</td>
<td>
An `integer` specifying the cutoff of how many items are considered in
the metric.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A `string` used as the name for this metric.
</td>
</tr><tr>
<td>
`gain_fn`
</td>
<td>
(function) Transforms labels. A method to calculate gain parameters
used in the definitions of the DCG and NDCG metrics, where the input is
the relevance label of the item. The gain is often defined to be of the
form 2^label-1.
</td>
</tr><tr>
<td>
`rank_discount_fn`
</td>
<td>
(function) The rank discount function. A method to define
the discount parameters used in the definitions of DCG and NDCG metrics,
where the input in the rank of item. The discount function is commonly
defined to be of the form log(rank+1).
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Other keyword arguments (e.g. alpha, seed).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A metric fn with the following Args:
* `labels`: A `Tensor` of the same shape as `predictions` representing
graded relevance.
* `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
is the ranking score of the corresponding example.
* `features`: A dict of `Tensor`s that contains all features.
</td>
</tr>

</table>
