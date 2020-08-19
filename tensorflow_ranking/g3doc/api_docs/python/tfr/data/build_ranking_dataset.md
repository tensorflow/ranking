<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_ranking_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_ranking_dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a ranking tf.dataset with a standard data format.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.data.build_ranking_dataset(
    file_pattern, data_format, batch_size, context_feature_spec,
    example_feature_spec, list_size=None, size_feature_name=None,
    shuffle_examples=False, seed=None, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`file_pattern`
</td>
<td>
See `build_ranking_dataset_with_parsing_fn`.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
See `make_parsing_fn`.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
See `build_ranking_dataset_with_parsing_fn`.
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
See `make_parsing_fn`.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
See `make_parsing_fn`.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
See `make_parsing_fn`.
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
</tr><tr>
<td>
`**kwargs`
</td>
<td>
The kwargs passed to `build_ranking_dataset_with_parsing_fn`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
See `build_ranking_dataset_with_parsing_fn`.
</td>
</tr>

</table>
