<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.data.parse_from_sequence_example" />
<meta itemprop="path" content="Stable" />
</div>

# ranking.data.parse_from_sequence_example

``` python
ranking.data.parse_from_sequence_example(
    serialized,
    list_size,
    context_feature_spec=None,
    example_feature_spec=None
)
```



Defined in [`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

Parses SequenceExample to feature maps.

#### Args:

* <b>`serialized`</b>: (Tensor) A string Tensor for a batch of serialized
    SequenceExample.
* <b>`list_size`</b>: (int) number of required frames in a SequenceExample. This is
    needed to normalize output tensor shapes across batches.
* <b>`context_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for context.
* <b>`example_feature_spec`</b>: (dict) A mapping from feature keys to
    `FixedLenFeature` or `VarLenFeature` values for the list of examples.
    These features are stored in the `feature_lists` field in SequenceExample.
    `FixedLenFeature` is translated to `FixedLenSequenceFeature` to parse
    SequenceExample. Note that no missing value in the middle of a
    `feature_list` is allowed for frames.


#### Returns:

A mapping from feature keys to `Tensor` or `SparseTensor`.