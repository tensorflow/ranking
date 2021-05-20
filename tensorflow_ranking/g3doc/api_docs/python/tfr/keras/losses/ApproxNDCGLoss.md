description: Computes approximate NDCG loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.ApproxNDCGLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.ApproxNDCGLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L769-L842">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes approximate NDCG loss between `y_true` and `y_pred`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.ApproxNDCGLoss(
    reduction=tf.losses.Reduction.AUTO, name=None, lambda_weight=None,
    temperature=0.1, ragged=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Implementation of ApproxNDCG loss ([Qin et al, 2008][qin2008];
[Bruch et al, 2019][bruch2019]). This loss is an approximation for
<a href="../../../tfr/keras/metrics/NDCGMetric.md"><code>tfr.keras.metrics.NDCGMetric</code></a>.
It replaces the non-differentiable ranking function in NDCG with a
differentiable approximation based on the logistic function.

For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

```
loss = sum_i (2^y_i - 1) / log_2(1 + approxrank(s_i))
approxrank(s_i) = 1 + sum_j (1 / (1 + exp(-(s_j - s_i) / temperature)))
```

#### Standalone usage:

```
>>> y_true = [[1., 0.]]
>>> y_pred = [[0.6, 0.8]]
>>> loss = tfr.keras.losses.ApproxNDCGLoss()
>>> loss(y_true, y_pred).numpy()
-0.6536734
```

```
>>> # Using ragged tensors
>>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
>>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
>>> loss = tfr.keras.losses.ApproxNDCGLoss(ragged=True)
>>> loss(y_true, y_pred).numpy()
-0.80536866
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd', loss=tfr.keras.losses.ApproxNDCGLoss())
```

#### Definition:

$$
\mathcal{L}(\{y\}, \{s\}) = - \frac{1}{\text{DCG}(y, y)} \sum_{i}
\frac{2^{y_i} - 1}{\log_2(1 + \text{rank}_i)}
$$

where:

$$
\text{rank}_i = 1 + \sum_{j \neq i}
\frac{1}{1 + \exp\left(\frac{-(s_j - s_i)}{\text{temperature}}\right)}
$$

#### References:

-   [A General Approximation Framework for Direct Optimization of Information
    Retrieval Measures, Qin et al, 2008][qin2008]
-   [Revisiting Approximate Metric Optimization in the Age of Deep Neural
    Networks, Bruch et al, 2019][bruch2019]

[qin2008]:
https://www.microsoft.com/en-us/research/publication/a-general-approximation-framework-for-direct-optimization-of-information-retrieval-measures/
[bruch2019]: https://research.google/pubs/pub48168/

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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L455-L463">View
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
