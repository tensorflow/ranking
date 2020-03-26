<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.ndcg" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.ndcg

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Computes NDCG from labels and ranks.

```python
tfr.utils.ndcg(
    labels, ranks=None, perm_mat=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`labels`</b>: A `Tensor` with shape [batch_size, list_size], representing
    graded relevance.
*   <b>`ranks`</b>: A `Tensor` of the same shape as labels, or [1, list_size],
    or None. If ranks=None, we assume the labels are sorted in their rank.
*   <b>`perm_mat`</b>: A `Tensor` with shape [batch_size, list_size, list_size]
    or None. Permutation matrices with rows correpond to the ranks and columns
    correspond to the indices. An argmax over each row gives the index of the
    element at the corresponding rank.

#### Returns:

A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
