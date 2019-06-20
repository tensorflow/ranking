<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_ranking_serving_input_receiver_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_ranking_serving_input_receiver_fn

Returns a serving input receiver fn for a standard data format.

```python
tfr.data.build_ranking_serving_input_receiver_fn(
    data_format,
    context_feature_spec,
    example_feature_spec,
    list_size=None,
    receiver_name='input_ranking_data',
    default_batch_size=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`data_format`</b>: (string) See RankingDataFormat.
*   <b>`context_feature_spec`</b>: (dict) Map from feature keys to
    `FixedLenFeature` or `VarLenFeature` values.
*   <b>`example_feature_spec`</b>: (dict) Map from feature keys to
    `FixedLenFeature` or `VarLenFeature` values.
*   <b>`list_size`</b>: (int) The number of examples to keep. If specified,
    truncation or padding may happen. Otherwise, set it to None to allow dynamic
    list size (recommended).
*   <b>`receiver_name`</b>: (string) The name for the receiver tensor.
*   <b>`default_batch_size`</b>: (int) Number of instances expected per batch.
    Leave unset for variable batch size (recommended).

#### Returns:

A `tf.estimator.export.ServingInputReceiver` object, which packages the
placeholders and the resulting feature Tensors together.
