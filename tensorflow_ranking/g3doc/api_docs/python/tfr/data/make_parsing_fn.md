<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.make_parsing_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.make_parsing_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a parsing fn for a standard data format.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.make_parsing_fn(
    data_format, list_size=None, context_feature_spec=None,
    example_feature_spec=None, size_feature_name=None, shuffle_examples=False,
    seed=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data_format`
</td>
<td>
(string) See RankingDataFormat.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
(int) The number of examples to keep per ranking instance. If
specified, truncation or padding may happen. Otherwise, the output Tensors
have a dynamic list size.
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature` or `VarLenFeature` values for context.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature` or `VarLenFeature` values for the list of examples.
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
`shuffle_examples`
</td>
<td>
(bool) A boolean to indicate whether examples within a
list are shuffled before the list is trimmed down to list_size elements
(when list has more than list_size elements).
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
(int) A seed passed onto random_ops.uniform() to shuffle examples.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A parsing function with signature parsing_fn(serialized), where serialized
is a string Tensor representing the serialized data in the specified
`data_format` and the function returns a feature map.
</td>
</tr>

</table>
