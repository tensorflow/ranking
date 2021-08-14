description: Factory method to create a metric based on a loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.make_loss_metric_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.losses.make_loss_metric_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py#L193-L268">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Factory method to create a metric based on a loss.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.losses.make_loss_metric_fn(
    loss_key, weights_feature_name=None, lambda_weight=None, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`loss_key`
</td>
<td>
A key in `RankingLossKey`.
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
`lambda_weight`
</td>
<td>
A `_LambdaWeight` object.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A `string` used as the name for this metric.
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
