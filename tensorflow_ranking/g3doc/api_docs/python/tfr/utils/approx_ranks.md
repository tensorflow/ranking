<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.approx_ranks" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.approx_ranks

Computes approximate ranks given a list of logits.

```python
tfr.utils.approx_ranks(
    logits,
    alpha=10.0
)
```

Defined in
[`python/utils.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py).

<!-- Placeholder for "Used in" -->

Given a list of logits, the rank of an item in the list is simply one plus the
total number of items with a larger logit. In other words,

rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

where "I" is the indicator function. The indicator function can be approximated
by a generalized sigmoid:

I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).

This function approximates the rank of an item using this sigmoid approximation
to the indicator function. This technique is at the core of "A general
approximation framework for direct optimization of information retrieval
measures" by Qin et al.

#### Args:

*   <b>`logits`</b>: A `Tensor` with shape [batch_size, list_size]. Each value
    is the ranking score of the corresponding item.
*   <b>`alpha`</b>: Exponent of the generalized sigmoid function.

#### Returns:

A `Tensor` of ranks with the same shape as logits.
