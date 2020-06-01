<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.inverse_max_dcg" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.inverse_max_dcg

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the inverse of max DCG.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.inverse_max_dcg(
    labels, gain_fn=(lambda labels: tf.pow(2.0, labels) - 1.0),
    rank_discount_fn=(lambda rank: 1.0 / tf.math.log1p(rank)), topn=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` with shape [batch_size, list_size]. Each value is the
graded relevance of the corresponding item.
</td>
</tr><tr>
<td>
`gain_fn`
</td>
<td>
A gain function. By default this is set to: 2^label - 1.
</td>
</tr><tr>
<td>
`rank_discount_fn`
</td>
<td>
A discount function. By default this is set to:
1/log(1+rank).
</td>
</tr><tr>
<td>
`topn`
</td>
<td>
An integer as the cutoff of examples in the sorted list.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="3">
A `Tensor` with shape [batch_size, 1].
</td>
</tr>

</table>
