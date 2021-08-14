description: Makes a loss function using a single loss or multiple losses.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.make_loss_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.losses.make_loss_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py#L38-L190">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Makes a loss function using a single loss or multiple losses.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.losses.make_loss_fn(
    loss_keys, loss_weights=None, weights_feature_name=None, lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, name=None,
    params=None, gumbel_params=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`loss_keys`
</td>
<td>
A string or list of strings representing loss keys defined in
`RankingLossKey`. Listed loss functions will be combined in a weighted
manner, with weights specified by `loss_weights`. If `loss_weights` is
None, default weight of 1 will be used.
</td>
</tr><tr>
<td>
`loss_weights`
</td>
<td>
List of weights, same length as `loss_keys`. Used when merging
losses to calculate the weighted sum of losses. If `None`, all losses are
weighted equally with weight being 1.
</td>
</tr><tr>
<td>
`weights_feature_name`
</td>
<td>
A string specifying the name of the weights feature in
`features` dict.
</td>
</tr><tr>
<td>
`lambda_weight`
</td>
<td>
A `_LambdaWeight` object created by factory methods like
`create_ndcg_lambda_weight()`.
</td>
</tr><tr>
<td>
`reduction`
</td>
<td>
One of `tf.losses.Reduction` except `NONE`. Describes how to
reduce training loss over batch.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string used as the name for this loss.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
A string-keyed dictionary that contains any other loss-specific
arguments.
</td>
</tr><tr>
<td>
`gumbel_params`
</td>
<td>
A string-keyed dictionary that contains other
`gumbel_softmax_sample` arguments.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A function _loss_fn(). See `_loss_fn()` for its signature.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `reduction` is invalid.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `loss_keys` is None or empty.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If `loss_keys` and `loss_weights` have different sizes.
</td>
</tr>
</table>
