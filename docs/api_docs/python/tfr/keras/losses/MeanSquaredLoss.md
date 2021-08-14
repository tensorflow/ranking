description: Computes mean squared loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.MeanSquaredLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.MeanSquaredLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L1074-L1122">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes mean squared loss between `y_true` and `y_pred`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.MeanSquaredLoss(
    reduction=tf.losses.Reduction.AUTO, name=None, ragged=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

```
loss = (y_true - y_pred)^2
```

#### Standalone usage:

```
>>> y_true = [[1., 0.]]
>>> y_pred = [[0.6, 0.8]]
>>> loss = tfr.keras.losses.MeanSquaredLoss()
>>> loss(y_true, y_pred).numpy()
0.4
```

```
>>> # Using ragged tensors
>>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
>>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
>>> loss = tfr.keras.losses.MeanSquaredLoss(ragged=True)
>>> loss(y_true, y_pred).numpy()
0.20833336
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd', loss=tfr.keras.losses.MeanSquaredLoss())
```

#### Definition:

$$
\mathcal{L}(\{y\}, \{s\}) = \sum_i (y_i - s_i)^{2}
$$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`reduction`
</td>
<td>
(Optional) The `tf.keras.losses.Reduction` to use (see
`tf.keras.losses.Loss`).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(Optional) The name for the op.
</td>
</tr><tr>
<td>
`ragged`
</td>
<td>
(Optional) If True, this loss will accept ragged tensors. If
False, this loss will accept dense tensors.
</td>
</tr>
</table>

## Methods

<h3 id="from_config"><code>from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L186-L191">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L172-L177">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true, y_pred, sample_weight=None
)
</code></pre>

See tf.keras.losses.Loss.
