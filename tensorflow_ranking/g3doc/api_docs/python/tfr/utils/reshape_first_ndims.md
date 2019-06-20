<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.reshape_first_ndims" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.reshape_first_ndims

Reshapes the first n dims of the input `tensor` to `new shape`.

```python
tfr.utils.reshape_first_ndims(
    tensor,
    first_ndims,
    new_shape
)
```

Defined in
[`python/utils.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`tensor`</b>: The input `Tensor`.
*   <b>`first_ndims`</b>: A int denoting the first n dims.
*   <b>`new_shape`</b>: A list of int representing the new shape.

#### Returns:

A reshaped `Tensor`.
