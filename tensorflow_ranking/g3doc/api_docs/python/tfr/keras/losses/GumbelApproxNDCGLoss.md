description: Computes the Gumbel approximate NDCG loss between y_true and
y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.GumbelApproxNDCGLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.GumbelApproxNDCGLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L846-L943">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the Gumbel approximate NDCG loss between `y_true` and `y_pred`.

Inherits From: [`ApproxNDCGLoss`](../../../tfr/keras/losses/ApproxNDCGLoss.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.GumbelApproxNDCGLoss(
    reduction=tf.losses.Reduction.AUTO, name=None, lambda_weight=None,
    temperature=0.1, sample_size=8, gumbel_temperature=1.0, seed=None, ragged=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Implementation of Gumbel ApproxNDCG loss ([Bruch et al, 2020][bruch2020]). This
loss is the same as
<a href="../../../tfr/keras/losses/ApproxNDCGLoss.md"><code>tfr.keras.losses.ApproxNDCGLoss</code></a>
but where logits are sampled from the Gumbel distribution:

`y_new_pred ~ Gumbel(y_pred, 1 / temperature)`

NOTE: This loss is stochastic and may return different values for identical
inputs.

#### Standalone usage:

```
>>> tf.random.set_seed(42)
>>> y_true = [[1., 0.]]
>>> y_pred = [[0.6, 0.8]]
>>> loss = tfr.keras.losses.GumbelApproxNDCGLoss(seed=42)
>>> loss(y_true, y_pred).numpy()
-0.8160851
```

```
>>> # Using a higher gumbel temperature
>>> loss = tfr.keras.losses.GumbelApproxNDCGLoss(gumbel_temperature=2.0,
...     seed=42)
>>> loss(y_true, y_pred).numpy()
-0.7583889
```

```
>>> # Using ragged tensors
>>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
>>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
>>> loss = tfr.keras.losses.GumbelApproxNDCGLoss(seed=42, ragged=True)
>>> loss(y_true, y_pred).numpy()
-0.69871885
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd', loss=tfr.keras.losses.GumbelApproxNDCGLoss())
```

#### Definition:

$$\mathcal{L}(\{y\}, \{s\}) = \text{ApproxNDCGLoss}(\{y\}, \{z\})$$

where

$$
z \sim \text{Gumbel}(s, \beta)\\
p(z) = e^{-t-e^{-t}}\\
t = \beta(z - s)\\
\beta = \frac{1}{\text{temperature}}
$$

#### References:

-   [A Stochastic Treatment of Learning to Rank Scoring Functions, Bruch et al,
    2020][bruch2020]

[bruch2020]: https://research.google/pubs/pub48689/

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`reduction`
</td>
<td>
(Optional) Type of `tf.keras.losses.Reduction` to apply to
loss. Default value is `AUTO`. `AUTO` indicates that the reduction
option will be determined by the usage context. For almost all cases
this defaults to `SUM_OVER_BATCH_SIZE`. When used with
`tf.distribute.Strategy`, outside of built-in training loops such as
`tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
will raise an error. Please see this custom training [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training) for
more details.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the op.
</td>
</tr>
</table>

## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L465-L472">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config, custom_objects=None
)
</code></pre>

Instantiates a `Loss` from its config (output of `get_config()`).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
Output of `get_config()`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Loss` instance.
</td>
</tr>

</table>

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L928-L935">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L937-L943">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true, y_pred, sample_weight=None
)
</code></pre>

See _RankingLoss.
