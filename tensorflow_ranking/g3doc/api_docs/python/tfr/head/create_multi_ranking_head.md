<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.head.create_multi_ranking_head" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.head.create_multi_ranking_head

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/head.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A factory method to create `_MultiRankingHead`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.head.create_multi_ranking_head(
    heads, head_weights=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`heads`
</td>
<td>
A tuple or list of `_RankingHead`.
</td>
</tr><tr>
<td>
`head_weights`
</td>
<td>
A tuple or list of weights.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance of `_MultiRankingHead` for multi-task learning.
</td>
</tr>

</table>
