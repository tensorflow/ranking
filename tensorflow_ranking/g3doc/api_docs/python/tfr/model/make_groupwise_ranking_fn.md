<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.model.make_groupwise_ranking_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.model.make_groupwise_ranking_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/model.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds an `Estimator` model_fn for groupwise comparison ranking models.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.model.make_groupwise_ranking_fn(
    group_score_fn, group_size, ranking_head, transform_fn=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`group_score_fn`
</td>
<td>
See `_GroupwiseRankingModel`.
</td>
</tr><tr>
<td>
`group_size`
</td>
<td>
See `_GroupwiseRankingModel`.
</td>
</tr><tr>
<td>
`ranking_head`
</td>
<td>
A `head._RankingHead` object.
</td>
</tr><tr>
<td>
`transform_fn`
</td>
<td>
See `_GroupwiseRankingModel`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
See `_make_model_fn`.
</td>
</tr>

</table>
