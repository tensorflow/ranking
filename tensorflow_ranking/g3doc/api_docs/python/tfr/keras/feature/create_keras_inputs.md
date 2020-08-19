<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.feature.create_keras_inputs" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.feature.create_keras_inputs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Create Keras inputs from context and example feature columns.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.feature.create_keras_inputs(
    context_feature_columns, example_feature_columns, size_feature_name,
    list_size=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`context_feature_columns`
</td>
<td>
(dict) context feature names to columns.
</td>
</tr><tr>
<td>
`example_feature_columns`
</td>
<td>
(dict) example feature names to columns.
</td>
</tr><tr>
<td>
`size_feature_name`
</td>
<td>
(str) Name of feature for example list sizes. If not
None, this feature name corresponds to a `tf.int32` Tensor of size
[batch_size] corresponding to sizes of example lists.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
(int) The list size for example features. If None, use dynamic
list size. A fixed list size is required for TPU training.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict mapping feature names to Keras Input tensors.
</td>
</tr>

</table>
