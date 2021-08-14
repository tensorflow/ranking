description: Factory method to get a ranking loss class.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.get" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.losses.get

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L28-L83">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Factory method to get a ranking loss class.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.get(
    loss, reduction=tf.losses.Reduction.AUTO, lambda_weight=None, name=None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`loss`
</td>
<td>
(str) An attribute of `RankingLossKey`, defining which loss object to
return.
</td>
</tr><tr>
<td>
`reduction`
</td>
<td>
(enum)  An enum of strings indicating the loss reduction type.
See type definition in the `tf.compat.v2.losses.Reduction`.
</td>
</tr><tr>
<td>
`lambda_weight`
</td>
<td>
(losses_impl._LambdaWeight) A lambda object for ranking
metric optimization.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(optional) (str) Name of loss.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments for the loss object.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A ranking loss instance. See `_RankingLoss` signature for more details.
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
If loss_key is unsupported.
</td>
</tr>
</table>
