<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.DCGLambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.losses.DCGLambdaWeight

## Class `DCGLambdaWeight`

LambdaWeight for Discounted Cumulative Gain metric.

Defined in
[`python/losses.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    topn=None,
    gain_fn=(lambda label: label),
    rank_discount_fn=(lambda rank: 1.0 / rank),
    normalized=False,
    smooth_fraction=0.0
)
```

Constructor.

Ranks are 1-based, not 0-based. Given rank i and j, there are two types of pair
weights: u = |rank_discount_fn(|i-j|) - rank_discount_fn(|i-j| + 1)| v =
|rank_discount_fn(i) - rank_discount_fn(j)| where u is the newly introduced one
in LambdaLoss paper (https://ai.google/research/pubs/pub47258) and v is the
original one in the LambdaMART paper "From RankNet to LambdaRank to LambdaMART:
An Overview". The final pair weight contribution of ranks is (1-smooth_fraction)
* u + smooth_fraction * v.

#### Args:

*   <b>`topn`</b>: (int) The topn for the DCG metric.
*   <b>`gain_fn`</b>: (function) Transforms labels.
*   <b>`rank_discount_fn`</b>: (function) The rank discount function.
*   <b>`normalized`</b>: (bool) If True, normalize weight by the max DCG.
*   <b>`smooth_fraction`</b>: (float) parameter to control the contribution from
    LambdaMART.

## Methods

<h3 id="individual_weights"><code>individual_weights</code></h3>

```python
individual_weights(sorted_labels)
```

See `_LambdaWeight`.

<h3 id="pair_weights"><code>pair_weights</code></h3>

```python
pair_weights(sorted_labels)
```

See `_LambdaWeight`.
