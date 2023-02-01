description: Computes approximate MRR loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.ApproxMRRLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.ApproxMRRLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L819-L887">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes approximate MRR loss between `y_true` and `y_pred`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.ApproxMRRLoss(
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    lambda_weight: Optional[losses_impl._LambdaWeight] = None,
    temperature: float = 0.1,
    ragged: bool = False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Implementation of ApproxMRR loss ([Qin et al, 2008][qin2008]). This loss is an
approximation for
<a href="../../../tfr/keras/metrics/MRRMetric.md"><code>tfr.keras.metrics.MRRMetric</code></a>.
It replaces the non-differentiable ranking function in MRR with a differentiable
approximation based on the logistic function.

For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

```
loss = sum_i (1 / approxrank(s_i)) * y_i
approxrank(s_i) = 1 + sum_j (1 / (1 + exp(-(s_j - s_i) / temperature)))
```

#### Standalone usage:

```
>>> y_true = [[1., 0.]]
>>> y_pred = [[0.6, 0.8]]
>>> loss = tfr.keras.losses.ApproxMRRLoss()
>>> loss(y_true, y_pred).numpy()
-0.53168947
```

```
>>> # Using ragged tensors
>>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
>>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
>>> loss = tfr.keras.losses.ApproxMRRLoss(ragged=True)
>>> loss(y_true, y_pred).numpy()
-0.73514676
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd', loss=tfr.keras.losses.ApproxMRRLoss())
```

#### Definition:

$$
\mathcal{L}(\{y\}, \{s\}) = -\sum_{i} \frac{1}{\text{approxrank}_i} y_i
$$

where:

$$
\text{approxrank}_i = 1 + \sum_{j \neq i}
\frac{1}{1 + \exp\left(\frac{-(s_j - s_i)}{\text{temperature}}\right)}
$$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
- [A General Approximation Framework for Direct Optimization of Information
Retrieval Measures, Qin et al, 2008][qin2008]
</td>
</tr>

</table>

[qin2008]: https://dl.acm.org/doi/10.1007/s10791-009-9124-x

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
loss. Default value is `AUTO`. `AUTO` indicates that the reduction
option will be determined by the usage context. For almost all cases
this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
`tf.distribute.Strategy`, except via `Model.compile()` and
`Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
will raise an error. Please see this custom training [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L585-L592">View
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L575-L583">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L221-L229">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    y_pred: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    sample_weight: Optional[utils.TensorLike] = None
) -> tf.Tensor
</code></pre>

See tf.keras.losses.Loss.
