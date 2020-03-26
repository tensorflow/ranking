<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.ordered_pair_accuracy" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.ordered_pair_accuracy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Computes the percentage of correctly ordered pair.

```python
tfr.metrics.ordered_pair_accuracy(
    labels, predictions, weights=None, name=None
)
```

<!-- Placeholder for "Used in" -->

For any pair of examples, we compare their orders determined by `labels` and
`predictions`. They are correctly ordered if the two orders are compatible. That
is, labels l_i > l_j and predictions s_i > s_j and the weight for this pair is
the weight from the l_i.

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*   <b>`predictions`</b>: A `Tensor` with shape [batch_size, list_size]. Each
    value is the ranking score of the corresponding example.
*   <b>`weights`</b>: A `Tensor` of the same shape of predictions or
    [batch_size, 1]. The former case is per-example and the latter case is
    per-list.
*   <b>`name`</b>: A string used as the name for this metric.

#### Returns:

A metric for the accuracy or ordered pairs.
