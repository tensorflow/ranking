<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.losses_impl.PrecisionLambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.losses.losses_impl.PrecisionLambdaWeight

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

## Class `PrecisionLambdaWeight`

<!-- Start diff -->

LambdaWeight for Precision metric.

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
__init__(
    topn,
    positive_fn=(lambda label: tf.greater_equal(label, 1.0))
)
```

Constructor.

#### Args:

*   <b>`topn`</b>: (int) The K in Precision@K metric.
*   <b>`positive_fn`</b>: (function): A function on `Tensor` that output boolean
    True for positive examples. The rest are negative examples.

## Methods

<h3 id="individual_weights"><code>individual_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
individual_weights(
    labels,
    ranks
)
```

Returns the weight `Tensor` for individual examples.

#### Args:

*   <b>`labels`</b>: A dense `Tensor` of labels with shape [batch_size,
    list_size].
*   <b>`ranks`</b>: A dense `Tensor` of ranks with the same shape as `labels`
    that are sorted by logits.

#### Returns:

A `Tensor` that can weight individual examples.

<h3 id="pair_weights"><code>pair_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
pair_weights(
    labels,
    ranks
)
```

See `_LambdaWeight`.

The current implementation here is that for any pairs of documents i and j, we
set the weight to be 1 if - i and j have different labels. - i <= topn and j >
topn or i > topn and j <= topn. This is exactly the same as the original
LambdaRank method. The weight is the gain of swapping a pair of documents.

#### Args:

*   <b>`labels`</b>: A dense `Tensor` of labels with shape [batch_size,
    list_size].
*   <b>`ranks`</b>: A dense `Tensor` of ranks with the same shape as `labels`
    that are sorted by logits.

#### Returns:

A `Tensor` that can weight example pairs.
