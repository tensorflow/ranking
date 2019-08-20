<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.build_ranking_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.build_ranking_dataset

Builds a ranking tf.dataset with a standard data format.

```python
tfr.data.build_ranking_dataset(
    file_pattern,
    data_format,
    batch_size,
    context_feature_spec,
    example_feature_spec,
    list_size=None,
    **kwargs
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`file_pattern`</b>: See `build_ranking_dataset_with_parsing_fn`.
*   <b>`data_format`</b>: See `make_parsing_fn`.
*   <b>`batch_size`</b>: See `build_ranking_dataset_with_parsing_fn`.
*   <b>`context_feature_spec`</b>: See `make_parsing_fn`.
*   <b>`example_feature_spec`</b>: See `make_parsing_fn`.
*   <b>`list_size`</b>: See `make_parsing_fn`.
*   <b>`**kwargs`</b>: The kwargs passed to
    `build_ranking_dataset_with_parsing_fn`.

#### Returns:

See `build_ranking_dataset_with_parsing_fn`.
