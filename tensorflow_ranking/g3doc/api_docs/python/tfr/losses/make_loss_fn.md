<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.make_loss_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.losses.make_loss_fn

Makes a loss function using a single loss or multiple losses.

```python
tfr.losses.make_loss_fn(
    loss_keys,
    loss_weights=None,
    weights_feature_name=None,
    lambda_weight=None,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    name=None,
    seed=None,
    extra_args=None
)
```

Defined in
[`python/losses.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`loss_keys`</b>: A string or list of strings representing loss keys
    defined in `RankingLossKey`. Listed loss functions will be combined in a
    weighted manner, with weights specified by `loss_weights`. If `loss_weights`
    is None, default weight of 1 will be used.
*   <b>`loss_weights`</b>: List of weights, same length as `loss_keys`. Used
    when merging losses to calculate the weighted sum of losses. If `None`, all
    losses are weighted equally with weight being 1.
*   <b>`weights_feature_name`</b>: A string specifying the name of the weights
    feature in `features` dict.
*   <b>`lambda_weight`</b>: A `_LambdaWeight` object created by factory methods
    like `create_ndcg_lambda_weight()`.
*   <b>`reduction`</b>: One of `tf.losses.Reduction` except `NONE`. Describes
    how to reduce training loss over batch.
*   <b>`name`</b>: A string used as the name for this loss.
*   <b>`seed`</b>: A randomization seed used in computation of some loss
    functions such as ListMLE and pListMLE.
*   <b>`extra_args`</b>: A string-keyed dictionary that contains any other
    loss-specific arguments.

#### Returns:

A function _loss_fn(). See `_loss_fn()` for its signature.

#### Raises:

*   <b>`ValueError`</b>: If `reduction` is invalid.
*   <b>`ValueError`</b>: If `loss_keys` is None or empty.
*   <b>`ValueError`</b>: If `loss_keys` and `loss_weights` have different sizes.
