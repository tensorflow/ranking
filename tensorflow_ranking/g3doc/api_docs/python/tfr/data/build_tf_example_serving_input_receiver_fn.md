description: Builds a serving input fn for tensorflow.training.Example.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_tf_example_serving_input_receiver_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_tf_example_serving_input_receiver_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py#L1332-L1369">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a serving input fn for `tensorflow.training.Example`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.build_tf_example_serving_input_receiver_fn(
    context_feature_spec, example_feature_spec, size_feature_name=None,
    mask_feature_name=None, default_batch_size=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) Map from feature keys to `FixedLenFeature`,
`VarLenFeature` or `RaggedFeature` values.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) Map from  feature keys to `FixedLenFeature`,
`VarLenFeature` or `RaggedFeature` values.
</td>
</tr><tr>
<td>
`size_feature_name`
</td>
<td>
(str) Name of feature for example list sizes. Populates
the feature dictionary with a `tf.int32` Tensor of value 1, and of shape
[batch_size] for this feature name. If None, which is default, this
feature is not generated.
</td>
</tr><tr>
<td>
`mask_feature_name`
</td>
<td>
(str) Name of feature for example list masks. Populates
the feature dictionary with a `tf.bool` Tensor of shape [batch_size,
list_size] for this feature name. If None, which is default, this feature
is not generated.
</td>
</tr><tr>
<td>
`default_batch_size`
</td>
<td>
(int) Number of instances expected per batch. Leave
unset for variable batch size (recommended).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `tf.estimator.export.ServingInputReceiver` object, which packages the
placeholders and the resulting feature Tensors together.
</td>
</tr>

</table>
