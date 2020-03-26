<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.average_relevance_position" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.average_relevance_position

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Computes average relevance position (ARP).

```python
tfr.metrics.average_relevance_position(
    labels, predictions, weights=None, name=None
)
```

<!-- Placeholder for "Used in" -->

This can also be named as average_relevance_rank, but this can be confusing with
mean_reciprocal_rank in acronyms. This name is more distinguishing and has been
used historically for binary relevance as average_click_position.

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*   <b>`predictions`</b>: A `Tensor` with shape [batch_size, list_size]. Each
    value is the ranking score of the corresponding example.
*   <b>`weights`</b>: A `Tensor` of the same shape of predictions or
    [batch_size, 1]. The former case is per-example and the latter case is
    per-list.
*   <b>`name`</b>: A string used as the name for this metric.

#### Returns:

A metric for the weighted average relevance position.
