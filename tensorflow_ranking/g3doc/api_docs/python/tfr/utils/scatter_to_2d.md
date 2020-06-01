<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.scatter_to_2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.scatter_to_2d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Scatters a flattened 1-D `tensor` to 2-D with padding based on `segments`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.scatter_to_2d(
    tensor, segments, pad_value, output_shape=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

For example: tensor = [1, 2, 3], segments = [0, 1, 0] and pad_value = -1, then
the returned 2-D tensor is [[1, 3], [2, -1]]. The output_shape is inferred when
None is provided. In this case, the shape will be dynamic and may not be
compatible with TPU. For TPU use case, please provide the `output_shape`
explicitly.

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A 1-D numeric `Tensor`.
</td>
</tr><tr>
<td>
`segments`
</td>
<td>
A 1-D int `Tensor` which is the idx output from tf.unique like [0,
0, 1, 0, 2]. See tf.unique. The segments may or may not be sorted.
</td>
</tr><tr>
<td>
`pad_value`
</td>
<td>
A numeric value to pad the output `Tensor`.
</td>
</tr><tr>
<td>
`output_shape`
</td>
<td>
A `Tensor` of size 2 telling the desired shape of the output
tensor. If None, the output_shape will be inferred and not fixed at
compilation time. When output_shape is smaller than needed, trucation will
be applied.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="properties responsive orange">
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="3">
A 2-D Tensor.
</td>
</tr>

</table>
