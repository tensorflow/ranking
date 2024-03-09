description: Computes Yeti logistic loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.YetiLogisticLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.YetiLogisticLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L594-L704">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes Yeti logistic loss between `y_true` and `y_pred`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.YetiLogisticLoss(
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    lambda_weight: Optional[<a href="../../../tfr/keras/losses/YetiDCGLambdaWeight.md"><code>tfr.keras.losses.YetiDCGLambdaWeight</code></a>] = None,
    temperature: float = 1.0,
    sample_size: int = 8,
    gumbel_temperature: float = 1.0,
    seed: Optional[int] = None,
    ragged: bool = False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Adapted to neural network models from the Yeti loss implemenation for GBDT in
([Lyzhin et al, 2022][lyzhin2022]).

In this code base, we support Yeti loss with the DCG lambda weight option. The
default uses the YetiDCGLambdaWeight with default settings. To customize, please
set the lambda_weight to YetiDCGLambdaWeight.

For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

```
loss = sum_a sum_i I[y_i > y_{i\pm 1}] * log(1 + exp(-(s^a_i - s^a_{i\pm 1})))
```

where `s^a_i = s_i + gumbel(0, 1)^a`

#### Standalone usage:

```
>>> y_true = [[1., 0.]]
>>> y_pred = [[0.6, 0.8]]
>>> loss = tfr.keras.losses.YetiLogisticLoss(sample_size=2, seed=1)
>>> loss(y_true, y_pred).numpy()
0.90761846
```

```
>>> # Using ragged tensors
>>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
>>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
>>> loss = tfr.keras.losses.YetiLogisticLoss(seed=1, ragged=True)
>>> loss(y_true, y_pred).numpy()
0.43420443
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd', loss=tfr.keras.losses.YetiLogisticLoss())
```

#### Definition:

$$
\mathcal{L}(\{y\}, \{s\}) =
\sum_a \sum_i \sum_{j=i\pm 1}I[y_i > y_j] \log(1 + \exp(-(s^a_i - s^a_j)))
$$

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
- [Which Tricks are Important for Learning to Rank?, Lyzhin et al, 2022][lyzhin2022]
</td>
</tr>

</table>

[lyzhin2022]: https://arxiv.org/abs/2204.01500

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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L298-L308">View
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L683-L690">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L692-L704">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    y_pred: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    sample_weight: Optional[utils.TensorLike] = None
) -> tf.Tensor
</code></pre>

See _RankingLoss.
