<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.padded_nd_indices" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.padded_nd_indices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Pads the invalid entries by valid ones and returns the nd_indices.

```python
tfr.utils.padded_nd_indices(
    is_valid, shuffle=False, seed=None
)
```

<!-- Placeholder for "Used in" -->

For example, when we have a batch_size = 1 and list_size = 3. Only the first 2
entries are valid. We have: `is_valid = [[True, True, False]] nd_indices, mask =
padded_nd_indices(is_valid)` nd_indices has a shape [1, 3, 2] and mask has a
shape [1, 3].

```
nd_indices = [[[0, 0], [0, 1], [0, 0]]]
mask = [[True, True, False]]
```

nd_indices can be used by gather_nd on a Tensor t `padded_t = tf.gather_nd(t,
nd_indices)` and get the following Tensor with first 2 dims are [1, 3]:
`padded_t = [[t(0, 0), t(0, 1), t(0, 0)]]`

#### Args:

*   <b>`is_valid`</b>: A boolean `Tensor` for entry validity with shape
    [batch_size, list_size].
*   <b>`shuffle`</b>: A boolean that indicates whether valid indices should be
    shuffled.
*   <b>`seed`</b>: Random seed for shuffle.

#### Returns:

A tuple of Tensors (nd_indices, mask). The first has shape [batch_size,
list_size, 2] and it can be used in gather_nd or scatter_nd. The second has the
shape of [batch_size, list_size] with value True for valid indices.
