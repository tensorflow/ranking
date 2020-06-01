<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.ndcg" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.ndcg

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes NDCG from labels and ranks.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.ndcg(
    labels, ranks=None, perm_mat=None
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
A `Tensor` with shape [batch_size, list_size], representing graded
relevance.
</td>
</tr><tr>
<td>
`ranks`
</td>
<td>
A `Tensor` of the same shape as labels, or [1, list_size], or None.
If ranks=None, we assume the labels are sorted in their rank.
</td>
</tr><tr>
<td>
`perm_mat`
</td>
<td>
A `Tensor` with shape [batch_size, list_size, list_size] or None.
Permutation matrices with rows correpond to the ranks and columns
correspond to the indices. An argmax over each row gives the index of the
element at the corresponding rank.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="3">
A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
</td>
</tr>

</table>
