<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.organize_valid_indices" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.organize_valid_indices

Organizes indices in such a way that valid items appear first.

```python
tfr.utils.organize_valid_indices(
    is_valid,
    shuffle=True,
    seed=None
)
```

Defined in
[`python/utils.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`is_valid`</b>: A boolen `Tensor` for entry validity with shape
    [batch_size, list_size].
*   <b>`shuffle`</b>: A boolean indicating whether valid items should be
    shuffled.
*   <b>`seed`</b>: An int for random seed at the op level. It works together
    with the seed at global graph level together to determine the random number
    generation. See `tf.set_random_seed`.

#### Returns:

A tensor of indices with shape [batch_size, list_size, 2]. The returned tensor
can be used with `tf.gather_nd` and `tf.scatter_nd` to compose a new
[batch_size, list_size] tensor. The values in the last dimension are the indices
for an element in the input tensor.
