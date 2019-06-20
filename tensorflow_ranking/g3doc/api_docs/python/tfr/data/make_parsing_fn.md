<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.make_parsing_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.make_parsing_fn

Returns a parsing fn for a standard data format.

```python
tfr.data.make_parsing_fn(
    data_format,
    list_size=None,
    context_feature_spec=None,
    example_feature_spec=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

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

#### Returns:

A parsing function with signature parsing_fn(serialized), where serialized is a
string Tensor representing the serialized data in the specified `data_format`
and the function returns a feature map.
