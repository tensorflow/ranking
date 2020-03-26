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
</td></table>

Builds a ranking tf.dataset with a standard data format.

```python
tfr.data.build_ranking_dataset(
    file_pattern, data_format, batch_size, context_feature_spec,
    example_feature_spec, list_size=None, size_feature_name=None,
    shuffle_examples=False, seed=None, **kwargs
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`file_pattern`</b>: See `build_ranking_dataset_with_parsing_fn`.
*   <b>`data_format`</b>: See `make_parsing_fn`.
*   <b>`batch_size`</b>: See `build_ranking_dataset_with_parsing_fn`.
*   <b>`context_feature_spec`</b>: See `make_parsing_fn`.
*   <b>`example_feature_spec`</b>: See `make_parsing_fn`.
*   <b>`list_size`</b>: See `make_parsing_fn`.
*   <b>`size_feature_name`</b>: (str) Name of feature for example list sizes.
    Populates the feature dictionary with a `tf.int32` Tensor of shape
    [batch_size] for this feature name. If None, which is default, this feature
    is not generated.
*   <b>`shuffle_examples`</b>: (bool) A boolean to indicate whether examples
    within a list are shuffled before the list is trimmed down to list_size
    elements (when list has more than list_size elements).
*   <b>`seed`</b>: (int) A seed passed onto random_ops.uniform() to shuffle
    examples.
*   <b>`**kwargs`</b>: The kwargs passed to
    `build_ranking_dataset_with_parsing_fn`.

#### Returns:

See `build_ranking_dataset_with_parsing_fn`.
