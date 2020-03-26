<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics.eval_metric" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.metrics.eval_metric

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

A stand-alone method to evaluate metrics on ranked results.

```python
tfr.metrics.eval_metric(
    metric_fn, **kwargs
)
```

<!-- Placeholder for "Used in" -->

Note that this method requires for the arguments of the metric to called
explicitly. So, the correct usage is of the following form:
tfr.metrics.eval_metric(tfr.metrics.mean_reciprocal_rank, labels=my_labels,
predictions=my_scores). Here is a simple example showing how to use this method:
import tensorflow_ranking as tfr scores = [[1., 3., 2.], [1., 2., 3.]] labels =
[[0., 0., 1.], [0., 1., 2.]] weights = [[1., 2., 3.], [4., 5., 6.]]
tfr.metrics.eval_metric( metric_fn=tfr.metrics.mean_reciprocal_rank,
labels=labels, predictions=scores, weights=weights) Args: metric_fn: (function)
Metric definition. A metric appearing in the TF-Ranking metrics module, e.g.
tfr.metrics.mean_reciprocal_rank **kwargs: A collection of argument values to be
passed to the metric, e.g. labels and predictions. See `_RankingMetric` and the
various metric definitions in tfr.metrics for the specifics.

#### Returns:

The evaluation of the metric on the input ranked lists.

#### Raises:

*   <b>`ValueError`</b>: One of the arguments required by the metric is not
    provided in the list of arguments included in kwargs.
