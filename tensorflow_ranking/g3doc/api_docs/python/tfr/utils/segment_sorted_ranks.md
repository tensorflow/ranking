<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.segment_sorted_ranks" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.segment_sorted_ranks

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns an int `Tensor` as the ranks after sorting scores per segment.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.segment_sorted_ranks(
    scores, segments, shuffle_ties=True, seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The returned ranks are 1-based. For example: scores = [1.0, 3.5, 2.1] segments =
[0, 0, 1] returned ranks = [2, 1, 1] The first 2 scores belong to the same
segment and the first score 1.0 is at rank 2 and second score 3.5 is in rank 1.
The last score is in another segment and its rank is 1 and there is no other
scores in this segment.

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scores`
</td>
<td>
A 1-D `Tensor` representing the scores to be sorted.
</td>
</tr><tr>
<td>
`segments`
</td>
<td>
A 1-D `Tensor` representing the segments that each score belongs
to. This should be the same shape as the scores.
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

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="3">
A 1-D int `Tensor`s as the ranks (1-based).
</td>
</tr>

</table>
