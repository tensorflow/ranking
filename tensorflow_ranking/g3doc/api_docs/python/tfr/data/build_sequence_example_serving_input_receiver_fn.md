<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_sequence_example_serving_input_receiver_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_sequence_example_serving_input_receiver_fn

Creates a serving_input_receiver_fn for `SequenceExample` inputs.

```python
tfr.data.build_sequence_example_serving_input_receiver_fn(
    input_size,
    context_feature_spec,
    example_feature_spec,
    default_batch_size=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

A string placeholder is used for inputs. Note that the context_feature_spec and
example_feature_spec shouldn't contain weights, labels or training only features
in general.

#### Args:

*   <b>`input_size`</b>: (int) The number of frames to keep in a
    SequenceExample. If specified, truncation or padding may happen. Otherwise,
    set it to None to allow dynamic list size (recommended).
*   <b>`context_feature_spec`</b>: (dict) Map from feature keys to
    `FixedLenFeature` or `VarLenFeature` values.
*   <b>`example_feature_spec`</b>: (dict) Map from feature keys to
    `FixedLenFeature` or `VarLenFeature` values.
*   <b>`default_batch_size`</b>: (int) Number of query examples expected per
    batch. Leave unset for variable batch size (recommended).

#### Returns:

A `tf.estimator.export.ServingInputReceiver` object, which packages the
placeholders and the resulting feature Tensors together.
