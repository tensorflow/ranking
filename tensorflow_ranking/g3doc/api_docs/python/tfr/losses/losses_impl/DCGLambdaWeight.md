<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.losses_impl.DCGLambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.losses.losses_impl.DCGLambdaWeight

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

## Class `DCGLambdaWeight`

<!-- Start diff -->

LambdaWeight for Discounted Cumulative Gain metric.

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
individual_weights(
    labels,
    ranks
)
```

See `_LambdaWeight`.

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
