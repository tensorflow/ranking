<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.data.build_sequence_example_serving_input_receiver_fn" />
<meta itemprop="path" content="Stable" />
</div>

# ranking.data.build_sequence_example_serving_input_receiver_fn

``` python
ranking.data.build_sequence_example_serving_input_receiver_fn(
    input_size,
    context_feature_spec,
    example_feature_spec,
    default_batch_size=None
)
```



Defined in [`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

Creates a serving_input_receiver_fn for `SequenceExample` inputs.

A string placeholder is used for inputs. Note that the context_feature_spec
and example_feature_spec shouldn't contain weights, labels or training
only features in general.

#### Args:

* <b>`input_size`</b>: (int) number of examples in an ExampleListWithContext. This is
    used for normalize SequenceExample.
* <b>`context_feature_spec`</b>: (dict) Map from feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
* <b>`example_feature_spec`</b>: (dict) Map from  feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
* <b>`default_batch_size`</b>: (int) Number of query examples expected per batch. Leave
    unset for variable batch size (recommended).


#### Returns:

A `tf.estimator.export.ServingInputReceiver` object, which packages the
placeholders and the resulting feature Tensors together.