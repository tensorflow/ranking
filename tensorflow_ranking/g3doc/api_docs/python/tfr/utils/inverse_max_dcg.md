<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.inverse_max_dcg" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.inverse_max_dcg

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Computes the inverse of max DCG.

```python
tfr.utils.inverse_max_dcg(
    labels,
    gain_fn=(lambda labels: tf.pow(2.0, labels) - 1.0),
    rank_discount_fn=(lambda rank: 1.0 / tf.math.log1p(rank)),
    topn=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`labels`</b>: A `Tensor` with shape [batch_size, list_size]. Each value
    is the graded relevance of the corresponding item.
*   <b>`gain_fn`</b>: A gain function. By default this is set to: 2^label - 1.
*   <b>`rank_discount_fn`</b>: A discount function. By default this is set to:
    1/log(1+rank).
*   <b>`topn`</b>: An integer as the cutoff of examples in the sorted list.

#### Returns:

A `Tensor` with shape [batch_size, 1].
