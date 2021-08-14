description: Creates _LambdaWeight based on Position-Aware ListMLE paper.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.create_p_list_mle_lambda_weight" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.losses.create_p_list_mle_lambda_weight

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py#L291-L305">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates _LambdaWeight based on Position-Aware ListMLE paper.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.losses.create_p_list_mle_lambda_weight(
    list_size
)
</code></pre>

<!-- Placeholder for "Used in" -->

Produces a weight based on the formulation presented in the "Position-Aware
ListMLE" paper (Lan et al.) and available using
create_p_list_mle_lambda_weight() factory function above.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`list_size`
</td>
<td>
Size of the input list.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A _LambdaWeight for Position-Aware ListMLE.
</td>
</tr>

</table>
