<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.PrecisionLambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.losses.PrecisionLambdaWeight

## Class `PrecisionLambdaWeight`

LambdaWeight for Precision metric.

Defined in
[`python/losses.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

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

```python
individual_weights(sorted_labels)
```

Returns the weight `Tensor` for individual examples.

#### Args:

*   <b>`sorted_labels`</b>: A dense `Tensor` of labels with shape [batch_size,
    list_size] that are sorted by logits.

#### Returns:

A `Tensor` that can weight individual examples.

<h3 id="pair_weights"><code>pair_weights</code></h3>

```python
pair_weights(sorted_labels)
```

See `_LambdaWeight`.

The current implementation here is that for any pairs of documents i and j, we
set the weight to be 1 if - i and j have different labels. - i <= topn and j >
topn or i > topn and j <= topn. This is exactly the same as the original
LambdaRank method. The weight is the gain of swapping a pair of documents.

#### Args:

*   <b>`sorted_labels`</b>: A dense `Tensor` of labels with shape [batch_size,
    list_size] that are sorted by logits.

#### Returns:

A `Tensor` that can weight example pairs.
