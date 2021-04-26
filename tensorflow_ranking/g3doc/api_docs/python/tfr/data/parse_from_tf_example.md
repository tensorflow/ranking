description: Parse function to convert tf.train.Example to feature maps.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.parse_from_tf_example" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.parse_from_tf_example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py#L1282-L1329">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Parse function to convert `tf.train.Example` to feature maps.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.parse_from_tf_example(
    serialized, context_feature_spec=None, example_feature_spec=None,
    size_feature_name=None, mask_feature_name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`serialized`
</td>
<td>
(`tf.train.Example`) A serialized proto object containing
context and example features.
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature`, `VarLenFeature` or `RaggedFeature` values for context
in `tf.train.Example` proto.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature`, `VarLenFeature` or `RaggedFeature` values for examples
in `tf.train.Example` proto.
</td>
</tr><tr>
<td>
`size_feature_name`
</td>
<td>
(str) Name of feature for example list sizes. Populates
the feature dictionary with a `tf.int32` Tensor of shape [batch_size] for
this feature name. If None, which is default, this feature is not
generated.
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
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A mapping from feature keys to `Tensor`, `SparseTensor` or `RaggedTensor`.
</td>
</tr>

</table>
