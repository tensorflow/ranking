<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.organize_valid_indices" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.organize_valid_indices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Organizes indices in such a way that valid items appear first.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.organize_valid_indices(
    is_valid, shuffle=True, seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`is_valid`
</td>
<td>
A boolean `Tensor` for entry validity with shape [batch_size,
list_size].
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
A boolean indicating whether valid items should be shuffled.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
An int for random seed at the op level. It works together with the
seed at global graph level together to determine the random number
generation. See `tf.set_random_seed`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of indices with shape [batch_size, list_size, 2]. The returned
tensor can be used with `tf.gather_nd` and `tf.scatter_nd` to compose a new
[batch_size, list_size] tensor. The values in the last dimension are the
indices for an element in the input tensor.
</td>
</tr>

</table>
