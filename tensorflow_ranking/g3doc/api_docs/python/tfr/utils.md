<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Utility functions for ranking library.

## Functions

[`de_noise(...)`](../tfr/utils/de_noise.md): Returns a float `Tensor` as the
de-noised `counts`.

[`is_label_valid(...)`](../tfr/utils/is_label_valid.md): Returns a boolean
`Tensor` for label validity.

[`organize_valid_indices(...)`](../tfr/utils/organize_valid_indices.md):
Organizes indices in such a way that valid items appear first.

[`padded_nd_indices(...)`](../tfr/utils/padded_nd_indices.md): Pads the invalid
entries by valid ones and returns the nd_indices.

[`reshape_first_ndims(...)`](../tfr/utils/reshape_first_ndims.md): Reshapes the
first n dims of the input `tensor` to `new shape`.

[`reshape_to_2d(...)`](../tfr/utils/reshape_to_2d.md): Converts the given
`tensor` to a 2-D `Tensor`.

[`shuffle_valid_indices(...)`](../tfr/utils/shuffle_valid_indices.md): Returns a
shuffle of indices with valid ones on top.

[`sort_by_scores(...)`](../tfr/utils/sort_by_scores.md): Sorts list of features
according to per-example scores.

[`sorted_ranks(...)`](../tfr/utils/sorted_ranks.md): Returns an int `Tensor` as
the ranks (1-based) after sorting scores.
