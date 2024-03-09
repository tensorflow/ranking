description: Computes calibrated softmax cross-entropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.CalibratedSoftmaxLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.CalibratedSoftmaxLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L821-L929">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes calibrated softmax cross-entropy loss.

Inherits From: [`SoftmaxLoss`](../../../tfr/keras/losses/SoftmaxLoss.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.CalibratedSoftmaxLoss(
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    lambda_weight: Optional[losses_impl._LambdaWeight] = None,
    temperature: float = 1.0,
    virtual_label: float = 0.0
)
</code></pre>

<!-- Placeholder for "Used in" -->

Implements calibrated softmax loss from ([Yan et al, 2022][yan2022]]).

Given virtual label `y_0`, for each list of scores `s` in `y_pred`, and list of
labels `y` in `y_true`:

```
loss = - sum_i y_i * log(exp(s_i) / (1 + sum_j exp(s_j)))
       - y_0 * log(1 / (1 + sum_j exp(s_j)))
```

#### Standalone usage:

```
>>> y_true = [[1., 0.]]
>>> y_pred = [[0.6, 0.8]]
>>> loss = tfr.keras.losses.CalibratedSoftmaxLoss(virtual_label=0.1)
>>> loss(y_true, y_pred).numpy()
1.1808171
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd', loss=tfr.keras.losses.CalibratedSoftmaxLoss())
```

#### Definition:

$$
\mathcal{L}(\{y\}, \{s\}) = - \sum_i y_i
\log\left(\frac{\exp(s_i)}{1+\sum_j \exp(s_j)}\right)
- y_0 \log\left(\frac{1}{1+\sum_j \exp(s_j)}\right)
$$

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
- [Scale Calibration of Deep Ranking Models, Yan et al, 2022][yan2022]
</td>
</tr>

</table>

[yan2022]: https://research.google/pubs/pub51402/

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`reduction`<a id="reduction"></a>
</td>
<td>
(Optional) The `tf.keras.losses.Reduction` to use (see
`tf.keras.losses.Loss`).
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
(Optional) The name for the op.
</td>
</tr><tr>
<td>
`lambda_weight`<a id="lambda_weight"></a>
</td>
<td>
(Optional) A lambdaweight to apply to the loss. Can be one
of <a href="../../../tfr/keras/losses/DCGLambdaWeight.md"><code>tfr.keras.losses.DCGLambdaWeight</code></a>,
<a href="../../../tfr/keras/losses/NDCGLambdaWeight.md"><code>tfr.keras.losses.NDCGLambdaWeight</code></a>, or,
<a href="../../../tfr/keras/losses/PrecisionLambdaWeight.md"><code>tfr.keras.losses.PrecisionLambdaWeight</code></a>.
</td>
</tr><tr>
<td>
`temperature`<a id="temperature"></a>
</td>
<td>
(Optional) The temperature to use for scaling the logits.
</td>
</tr><tr>
<td>
`virtual_label`<a id="virtual_label"></a>
</td>
<td>
A NON-NEGATIVE float number as the hyperparameter of the
loss to control the relative scores to zero of results with labels.
</td>
</tr>
</table>

## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L731-L741">View
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L888-L893">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L895-L929">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    y_pred: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    sample_weight: Optional[utils.TensorLike] = None
) -> tf.Tensor
</code></pre>

See _RankingLoss.
