<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.data.libsvm_generator" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.data.libsvm_generator

Parses a LibSVM-formatted input file and aggregates data points by qid.

```python
tfr.data.libsvm_generator(
    path,
    num_features,
    list_size,
    seed=None
)
```

Defined in
[`python/data.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/data.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`path`</b>: (string) path to dataset in the LibSVM format.
*   <b>`num_features`</b>: An integer representing the number of features per
    instance.
*   <b>`list_size`</b>: Size of the document list per query.
*   <b>`seed`</b>: Randomization seed used when shuffling the document list.

#### Returns:

A generator function that can be passed to tf.data.Dataset.from_generator().
