<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.head.create_ranking_head" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.head.create_ranking_head

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/head.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

<!-- Start diff -->

A factory method to create `_RankingHead`.

```python
tfr.head.create_ranking_head(
    loss_fn,
    eval_metric_fns=None,
    optimizer=None,
    train_op_fn=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`loss_fn`</b>: A loss function with the following signature (see
    make_loss_fn in losses.py):
    *   Args: `labels`: A `Tensor` of the same shape as `logits` representing
        relevance. `logits`: A `Tensor` with shape [batch_size, list_size]. Each
        value is the ranking score of the corresponding example. `features`: A
        dict of `Tensor`s for all features.
    *   Returns: A scalar containing the loss to be optimized.
*   <b>`eval_metric_fns`</b>: A dict of metric functions keyed by a string name.
    The values of the dict are metric functions with the following signature:
    *   Args: `labels`: A `Tensor` of the same shape as `predictions`
        representing relevance. `predictions`: A `Tensor` with shape
        [batch_size, list_size]. Each value is the ranking score of the
        corresponding example. `features`: A dict of `Tensor`s for all features.
    *   Returns: The result of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple.
*   <b>`optimizer`</b>: `Optimizer` instance used to optimize the loss in TRAIN
    mode. Namely, it sets `train_op = optimizer.minimize(loss, global_step)`,
    which updates variables and increments `global_step`.
*   <b>`train_op_fn`</b>: Function that takes a scalar loss `Tensor` and returns
    `train_op`. Used if `optimizer` is `None`.
*   <b>`name`</b>: Name of the head. If provided, will be used as `name_scope`
    when creating ops.

#### Returns:

An instance of `_RankingHead` for ranking.

#### Raises:

*   <b>`ValueError`</b>: If `loss_fn` is not callable.
