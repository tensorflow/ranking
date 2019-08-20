<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.utils

Utility functions for ranking library.

Defined in
[`python/utils.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py).

<!-- Placeholder for "Used in" -->

## Functions

[`approx_ranks(...)`](../tfr/utils/approx_ranks.md): Computes approximate ranks
given a list of logits.

[`inverse_max_dcg(...)`](../tfr/utils/inverse_max_dcg.md): Computes the inverse
of max DCG.

[`is_label_valid(...)`](../tfr/utils/is_label_valid.md): Returns a boolean
`Tensor` for label validity.

[`organize_valid_indices(...)`](../tfr/utils/organize_valid_indices.md):
Organizes indices in such a way that valid items appear first.

[`reshape_first_ndims(...)`](../tfr/utils/reshape_first_ndims.md): Reshapes the
first n dims of the input `tensor` to `new shape`.

[`reshape_to_2d(...)`](../tfr/utils/reshape_to_2d.md): Converts the given
`tensor` to a 2-D `Tensor`.

[`shuffle_valid_indices(...)`](../tfr/utils/shuffle_valid_indices.md): Returns a
shuffle of indices with valid ones on top.

[`sort_by_scores(...)`](../tfr/utils/sort_by_scores.md): Sorts example features
according to per-example scores.
