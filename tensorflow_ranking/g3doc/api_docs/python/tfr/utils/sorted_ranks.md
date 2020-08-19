<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.sorted_ranks" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.sorted_ranks

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns an int `Tensor` as the ranks (1-based) after sorting scores.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.sorted_ranks(
    scores, shuffle_ties=True, seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Example: Given scores = [[1.0, 3.5, 2.1]], the returned ranks will be [[3, 1,
2]]. It means that scores 1.0 will be ranked at position 3, 3.5 will be ranked
at position 1, and 2.1 will be ranked at position 2.

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
`shuffle_ties`
</td>
<td>
See `sort_by_scores`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
See `sort_by_scores`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 1-based int `Tensor`s as the ranks.
</td>
</tr>

</table>
