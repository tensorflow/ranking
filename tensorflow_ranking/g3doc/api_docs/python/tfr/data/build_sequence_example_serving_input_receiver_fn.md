<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_sequence_example_serving_input_receiver_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_sequence_example_serving_input_receiver_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates a serving_input_receiver_fn for `SequenceExample` inputs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.build_sequence_example_serving_input_receiver_fn(
    input_size, context_feature_spec, example_feature_spec, default_batch_size=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

A string placeholder is used for inputs. Note that the context_feature_spec and
example_feature_spec shouldn't contain weights, labels or training only features
in general.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_size`
</td>
<td>
(int) The number of frames to keep in a SequenceExample. If
specified, truncation or padding may happen. Otherwise, set it to None to
allow dynamic list size (recommended).
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) Map from feature keys to `FixedLenFeature` or
`VarLenFeature` values.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) Map from  feature keys to `FixedLenFeature` or
`VarLenFeature` values.
</td>
</tr><tr>
<td>
`default_batch_size`
</td>
<td>
(int) Number of query examples expected per batch. Leave
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
