<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_ranking_serving_input_receiver_fn_with_parsing_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_ranking_serving_input_receiver_fn_with_parsing_fn

Returns a receiver function with the provided `parsing_fn`.

```python
tfr.data.build_ranking_serving_input_receiver_fn_with_parsing_fn(
    parsing_fn,
    receiver_name,
    default_batch_size=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`parsing_fn`</b>: (function) It has a single argument
    parsing_fn(serialized). Users can customize this for their own data formats.
*   <b>`receiver_name`</b>: (string) The name for the reveiver Tensor that
    contains the serialized data.
*   <b>`default_batch_size`</b>: (int) Number of instances expected per batch.
    Leave unset for variable batch size (recommended).

#### Returns:

A `tf.estimator.export.ServingInputReceiver` object, which packages the
placeholders and the resulting feature Tensors together.
