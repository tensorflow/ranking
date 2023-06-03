description: Computes the Ordinal loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.OrdinalLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.OrdinalLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L1475-L1528">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the Ordinal loss between `y_true` and `y_pred`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.OrdinalLoss(
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    ragged: bool = False,
    ordinal_size: int = 1,
    use_fraction_label: bool = False
)
</code></pre>

<!-- Placeholder for "Used in" -->

In ordinal loss, y_pred is a 3D tensor with the last dimension equals to
ordinal_size. `loss = -\sum_i=0^ordinal_size-1 I_{y_true > i}
log(sigmoid(y_pred[i])) + I_{y_true <= i} log(1-sigmoid(y_pred[i]))`

#### Standalone usage:

```
>>> y_true = [[1., 0.]]
>>> y_pred = [[[0.6, 0.2], [0.8, 0.3]]]
>>> loss = tfr.keras.losses.OrdinalLoss(ordinal_size=2)
>>> loss(y_true, y_pred).numpy()
1.6305413
```

```
>>> # Using ragged tensors
>>> y_true = tf.ragged.constant([[2., 1.], [0.]])
>>> y_pred = tf.ragged.constant([[[0.6, 0.2], [0.8, 0.3]], [[0., -0.2]]])
>>> loss = tfr.keras.losses.OrdinalLoss(ordinal_size=2, ragged=True)
>>> loss(y_true, y_pred).numpy()
0.88809216
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd',
              loss=tfr.keras.losses.OrdinalLoss(ordinal_size=2))
```

#### Definition:

$$
\mathcal{L}(\{y\}, \{s\}) = - \sum_i\sum_{j=0}^{m-1} I_{y_i > j}
    \log(\text{sigmoid}(s_{i,j})) + I_{y_i \leq j}
    \log(1 - \text{sigmoid}(s_{i,j}))
$$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`reduction`<a id="reduction"></a>
</td>
<td>
Type of `tf.keras.losses.Reduction` to apply to
loss. Default value is `AUTO`. `AUTO` indicates that the
reduction option will be determined by the usage context. For
almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
used under a `tf.distribute.Strategy`, except via
`Model.compile()` and `Model.fit()`, using `AUTO` or
`SUM_OVER_BATCH_SIZE` will raise an error. Please see this
custom training [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)
for more details.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the instance.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L266-L269">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L248-L256">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    y_pred: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    sample_weight: Optional[utils.TensorLike] = None
) -> tf.Tensor
</code></pre>

See tf.keras.losses.Loss.
