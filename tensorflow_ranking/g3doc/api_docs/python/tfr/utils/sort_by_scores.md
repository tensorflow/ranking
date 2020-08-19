<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.sort_by_scores" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.sort_by_scores

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Sorts list of features according to per-example scores.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.sort_by_scores(
    scores, features_list, topn=None, shuffle_ties=True, seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scores`
</td>
<td>
A `Tensor` of shape [batch_size, list_size] representing the
per-example scores.
</td>
</tr><tr>
<td>
`features_list`
</td>
<td>
A list of `Tensor`s to be sorted. The shape of the `Tensor`
can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
The latter is applicable for example features.
</td>
</tr><tr>
<td>
`topn`
</td>
<td>
An integer as the cutoff of examples in the sorted list.
</td>
</tr><tr>
<td>
`shuffle_ties`
</td>
<td>
A boolean. If True, randomly shuffle before the sorting.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
The ops-level random seed used when `shuffle_ties` is True.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of `Tensor`s as the list of sorted features by `scores`.
</td>
</tr>

</table>
