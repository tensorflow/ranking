<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.reshape_first_ndims" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.reshape_first_ndims

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reshapes the first n dims of the input `tensor` to `new shape`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.reshape_first_ndims(
    tensor, first_ndims, new_shape
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
The input `Tensor`.
</td>
</tr><tr>
<td>
`first_ndims`
</td>
<td>
A int denoting the first n dims.
</td>
</tr><tr>
<td>
`new_shape`
</td>
<td>
A list of int representing the new shape.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A reshaped `Tensor`.
</td>
</tr>

</table>
