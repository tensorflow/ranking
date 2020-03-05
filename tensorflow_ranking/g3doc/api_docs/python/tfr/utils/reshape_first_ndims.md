<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.reshape_first_ndims" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.reshape_first_ndims

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Reshapes the first n dims of the input `tensor` to `new shape`.

```python
tfr.utils.reshape_first_ndims(
    tensor, first_ndims, new_shape
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`tensor`</b>: The input `Tensor`.
*   <b>`first_ndims`</b>: A int denoting the first n dims.
*   <b>`new_shape`</b>: A list of int representing the new shape.

#### Returns:

A reshaped `Tensor`.
