<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.make_loss_metric_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.losses.make_loss_metric_fn

Factory method to create a metric based on a loss.

```python
tfr.losses.make_loss_metric_fn(
    loss_key,
    weights_feature_name=None,
    lambda_weight=None,
    name=None
)
```

Defined in
[`python/losses.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`loss_key`</b>: A key in `RankingLossKey`.
*   <b>`weights_feature_name`</b>: A `string` specifying the name of the weights
    feature in `features` dict.
*   <b>`lambda_weight`</b>: A `_LambdaWeight` object.
*   <b>`name`</b>: A `string` used as the name for this metric.

#### Returns:

A metric fn with the following Args: * `labels`: A `Tensor` of the same shape as
`predictions` representing graded relevance. * `predictions`: A `Tensor` with
shape [batch_size, list_size]. Each value is the ranking score of the
corresponding example. * `features`: A dict of `Tensor`s that contains all
features.
