description: Computes alpha discounted cumulative gain (alpha-DCG).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.alpha_discounted_cumulative_gain" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.alpha_discounted_cumulative_gain

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py#L495-L538">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes alpha discounted cumulative gain (alpha-DCG).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.metrics.alpha_discounted_cumulative_gain(
    labels, predictions, weights=None, topn=None, name=None,
    rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN, alpha=0.5, seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`
</td>
<td>
A `Tensor` with shape [batch_size, list_size, subtopic_size]. Each
value represents graded relevance to a subtopic: 1 for relevent subtopic,
0 for irrelevant, and -1 for paddings. When the actual subtopic number
of a query is smaller than the `subtopic_size`, `labels` will be padded
to `subtopic_size` with -1, similar to the paddings used for queries
with doc number less then list_size.
</td>
</tr><tr>
<td>
`predictions`
</td>
<td>
A `Tensor` with shape [batch_size, list_size]. Each value is
the ranking score of the corresponding example.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
A `Tensor` of shape [batch_size, list_size] or [batch_size, 1].
They are per-example and per-list, respectively.
</td>
</tr><tr>
<td>
`topn`
</td>
<td>
A cutoff for how many examples to consider for this metric.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A string used as the name for this metric.
</td>
</tr><tr>
<td>
`rank_discount_fn`
</td>
<td>
A function of rank discounts. Default is set to
discount = 1 / log2(rank+1).
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
A float between 0 and 1. Originally introduced as an assessor error
in judging whether a document is covering a subtopic of the query. It
can also be interpreted as the inverse number of documents covering the
same subtopic reader needs to get and confirm the subtopic information
of a query.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
The ops-level random seed used in shuffle ties in `sort_by_scores`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A metric for the weighted alpha discounted cumulative gain of the batch.
</td>
</tr>

</table>
