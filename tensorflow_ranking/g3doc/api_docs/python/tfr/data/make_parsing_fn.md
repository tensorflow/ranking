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
</td></table>

Returns a parsing fn for a standard data format.

```python
tfr.data.make_parsing_fn(
    data_format, list_size=None, context_feature_spec=None,
    example_feature_spec=None, size_feature_name=None, shuffle_examples=False,
    seed=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`data_format`</b>: (string) See RankingDataFormat.
*   <b>`list_size`</b>: (int) The number of examples to keep per ranking
    instance. If specified, truncation or padding may happen. Otherwise, the
    output Tensors have a dynamic list size.
*   <b>`context_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for context.
*   <b>`example_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for the list of examples.
*   <b>`size_feature_name`</b>: (str) Name of feature for example list sizes.
    Populates the feature dictionary with a `tf.int32` Tensor of shape
    [batch_size] for this feature name. If None, which is default, this feature
    is not generated.
*   <b>`shuffle_examples`</b>: (bool) A boolean to indicate whether examples
    within a list are shuffled before the list is trimmed down to list_size
    elements (when list has more than list_size elements).
*   <b>`seed`</b>: (int) A seed passed onto random_ops.uniform() to shuffle
    examples.

#### Returns:

A parsing function with signature parsing_fn(serialized), where serialized is a
string Tensor representing the serialized data in the specified `data_format`
and the function returns a feature map.
