<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.losses_impl.ApproxMRRLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compute"/>
<meta itemprop="property" content="compute_unreduced_loss"/>
<meta itemprop="property" content="eval_metric"/>
<meta itemprop="property" content="normalize_weights"/>
</div>

# tfr.losses.losses_impl.ApproxMRRLoss

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

## Class `ApproxMRRLoss`

<!-- Start diff -->

Implements ApproxMRR loss.

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
__init__(
    name,
    lambda_weight=None,
    params=None
)
```

Constructor.

#### Args:

*   <b>`name`</b>: A string used as the name for this loss.
*   <b>`lambda_weight`</b>: A `_LambdaWeight` object.
*   <b>`params`</b>: A dict for params used in loss computation.

## Properties

<h3 id="name"><code>name</code></h3>

The loss name.

## Methods

<h3 id="compute"><code>compute</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
compute(
    labels,
    logits,
    weights,
    reduction
)
```

Computes the reduced loss for tf.estimator (not tf.keras).

Note that this function is not compatible with keras.

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `logits` representing
    graded relevance.
*   <b>`logits`</b>: A `Tensor` with shape [batch_size, list_size]. Each value
    is the ranking score of the corresponding item.
*   <b>`weights`</b>: A scalar, a `Tensor` with shape [batch_size, 1] for
    list-wise weights, or a `Tensor` with shape [batch_size, list_size] for
    item-wise weights.
*   <b>`reduction`</b>: One of `tf.losses.Reduction` except `NONE`. Describes
    how to reduce training loss over batch.

#### Returns:

Reduced loss for training and eval.

<h3 id="compute_unreduced_loss"><code>compute_unreduced_loss</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
compute_unreduced_loss(
    labels,
    logits
)
```

See `_RankingLoss`.

<h3 id="eval_metric"><code>eval_metric</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
eval_metric(
    labels,
    logits,
    weights
)
```

Computes the eval metric for the loss in tf.estimator (not tf.keras).

Note that this function is not compatible with keras.

#### Args:

*   <b>`labels`</b>: A `Tensor` of the same shape as `logits` representing
    graded relevance.
*   <b>`logits`</b>: A `Tensor` with shape [batch_size, list_size]. Each value
    is the ranking score of the corresponding item.
*   <b>`weights`</b>: A scalar, a `Tensor` with shape [batch_size, 1] for
    list-wise weights, or a `Tensor` with shape [batch_size, list_size] for
    item-wise weights.

#### Returns:

A metric op.

<h3 id="normalize_weights"><code>normalize_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
normalize_weights(
    labels,
    weights
)
```

Normalizes weights needed for tf.estimator (not tf.keras).

This is needed for `tf.estimator` given that the reduction may be
`SUM_OVER_NONZERO_WEIGHTS`. This function is not needed after we migrate from
the deprecated reduction to `SUM` or `SUM_OVER_BATCH_SIZE`.

#### Args:

*   <b>`labels`</b>: A `Tensor` of shape [batch_size, list_size] representing
    graded relevance.
*   <b>`weights`</b>: A scalar, a `Tensor` with shape [batch_size, 1] for
    list-wise weights, or a `Tensor` with shape [batch_size, list_size] for
    item-wise weights.

#### Returns:

The normalized weights.
