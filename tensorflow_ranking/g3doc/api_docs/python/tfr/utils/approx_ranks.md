<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.approx_ranks" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.approx_ranks

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes approximate ranks given a list of logits.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.approx_ranks(
    logits, alpha=10.0
)
</code></pre>

<!-- Placeholder for "Used in" -->

Given a list of logits, the rank of an item in the list is simply one plus the
total number of items with a larger logit. In other words,

rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

where "I" is the indicator function. The indicator function can be approximated
by a generalized sigmoid:

I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).

This function approximates the rank of an item using this sigmoid approximation
to the indicator function. This technique is at the core of "A general
approximation framework for direct optimization of information retrieval
measures" by Qin et al.

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`logits`
</td>
<td>
A `Tensor` with shape [batch_size, list_size]. Each value is the
ranking score of the corresponding item.
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
Exponent of the generalized sigmoid function.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="3">
A `Tensor` of ranks with the same shape as logits.
</td>
</tr>

</table>
