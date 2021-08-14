description: Keras serializable class for Precision.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.PrecisionLambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.keras.losses.PrecisionLambdaWeight

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L131-L142">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Keras serializable class for Precision.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.PrecisionLambdaWeight(
    topn=None, positive_fn=None, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`topn`
</td>
<td>
(int) The K in Precision@K metric.
</td>
</tr><tr>
<td>
`positive_fn`
</td>
<td>
(function): A function on `Tensor` that output boolean True
for positive examples. The rest are negative examples.
</td>
</tr>
</table>

## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L138-L142">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

<h3 id="individual_weights"><code>individual_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L163-L175">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>individual_weights(
    labels, ranks
)
</code></pre>

Returns the weight `Tensor` for individual examples.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`labels`
</td>
<td>
A dense `Tensor` of labels with shape [batch_size, list_size].
</td>
</tr><tr>
<td>
`ranks`
</td>
<td>
A dense `Tensor` of ranks with the same shape as `labels` that are
sorted by logits.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` that can weight individual examples.
</td>
</tr>

</table>

<h3 id="pair_weights"><code>pair_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L309-L337">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pair_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.

The current implementation here is that for any pairs of documents i and j, we
set the weight to be 1 if - i and j have different labels. - i <= topn and j >
topn or i > topn and j <= topn. This is exactly the same as the original
LambdaRank method. The weight is the gain of swapping a pair of documents.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`labels`
</td>
<td>
A dense `Tensor` of labels with shape [batch_size, list_size].
</td>
</tr><tr>
<td>
`ranks`
</td>
<td>
A dense `Tensor` of ranks with the same shape as `labels` that are
sorted by logits.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` that can weight example pairs.
</td>
</tr>

</table>
