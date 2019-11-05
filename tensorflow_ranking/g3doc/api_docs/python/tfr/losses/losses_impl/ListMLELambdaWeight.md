<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses.losses_impl.ListMLELambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.losses.losses_impl.ListMLELambdaWeight

<!-- Insert buttons -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

## Class `ListMLELambdaWeight`

<!-- Start diff -->

LambdaWeight for ListMLE cost function.

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py">View
source</a>

```python
__init__(rank_discount_fn)
```

Constructor.

Ranks are 1-based, not 0-based.

#### Args:

*   <b>`rank_discount_fn`</b>: (function) The rank discount function.

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
