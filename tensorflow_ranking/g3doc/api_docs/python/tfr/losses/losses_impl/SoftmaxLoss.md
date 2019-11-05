<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.losses_impl.SoftmaxLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compute"/>
<meta itemprop="property" content="compute_unreduced_loss"/>
<meta itemprop="property" content="eval_metric"/>
<meta itemprop="property" content="normalize_weights"/>
<meta itemprop="property" content="precompute"/>
</div>

# tfr.losses.losses_impl.SoftmaxLoss

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

## Class `SoftmaxLoss`

<!-- Start diff -->

Implements softmax loss.

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

See `_RankingLoss`.

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

See `_RankingLoss`.

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

<h3 id="precompute"><code>precompute</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
precompute(
    labels,
    logits,
    weights
)
```

Precomputes Tensors for softmax cross entropy inputs.
