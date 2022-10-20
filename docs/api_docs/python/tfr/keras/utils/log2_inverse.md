description: Computes 1./log2(1+x) element-wise for each label.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.utils.log2_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.utils.log2_inverse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/utils.py#L60-L73">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes `1./log2(1+x)` element-wise for each label.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.utils.log2_inverse(
    rank: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Can be used to define `rank_discount_fn` for
<a href="../../../tfr/keras/metrics/NDCGMetric.md"><code>tfr.keras.metrics.NDCGMetric</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`rank`<a id="rank"></a>
</td>
<td>
A `Tensor` or anything that can be converted to a tensor using
`tf.convert_to_tensor`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` that has each input element transformed as `x` to `1./log2(1+x)`.
</td>
</tr>

</table>
