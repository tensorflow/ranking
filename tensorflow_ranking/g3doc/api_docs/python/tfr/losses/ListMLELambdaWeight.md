<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.ListMLELambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.losses.ListMLELambdaWeight

## Class `ListMLELambdaWeight`

LambdaWeight for ListMLE cost function.

Defined in
[`python/losses.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(rank_discount_fn)
```

Constructor.

Ranks are 1-based, not 0-based.

#### Args:

*   <b>`rank_discount_fn`</b>: (function) The rank discount function.

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
