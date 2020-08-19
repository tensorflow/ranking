<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.head.create_ranking_head" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.head.create_ranking_head

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/head.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A factory method to create `_RankingHead`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.head.create_ranking_head(
    loss_fn, eval_metric_fns=None, optimizer=None, train_op_fn=None, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`loss_fn`
</td>
<td>
A loss function with the following signature (see make_loss_fn in
losses.py):
* Args:
`labels`: A `Tensor` of the same shape as `logits` representing
relevance.
`logits`: A `Tensor` with shape [batch_size, list_size]. Each value is
the ranking score of the corresponding example.
`features`: A dict of `Tensor`s for all features.
* Returns: A scalar containing the loss to be optimized.
</td>
</tr><tr>
<td>
`eval_metric_fns`
</td>
<td>
A dict of metric functions keyed by a string name. The
values of the dict are metric functions with the following signature:
* Args:
`labels`: A `Tensor` of the same shape as `predictions` representing
relevance.
`predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
is the ranking score of the corresponding example.
`features`: A dict of `Tensor`s for all features.
* Returns: The result of calling a metric function, namely a
`(metric_tensor, update_op)` tuple.
</td>
</tr><tr>
<td>
`optimizer`
</td>
<td>
`Optimizer` instance used to optimize the loss in TRAIN mode.
Namely, it sets `train_op = optimizer.minimize(loss, global_step)`, which
updates variables and increments `global_step`.
</td>
</tr><tr>
<td>
`train_op_fn`
</td>
<td>
Function that takes a scalar loss `Tensor` and returns
`train_op`. Used if `optimizer` is `None`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name of the head. If provided, will be used as `name_scope` when
creating ops.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance of `_RankingHead` for ranking.
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
If `loss_fn` is not callable.
</td>
</tr>
</table>
