description: Softmax cross-entropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.SoftmaxLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.SoftmaxLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L388-L429">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Softmax cross-entropy loss.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.SoftmaxLoss(
    reduction=tf.losses.Reduction.AUTO, name=None, lambda_weight=None,
    temperature=1.0, ragged=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

$$
\mathcal{L}(\{y\}, \{s\}) =
- \sum_i \frac{y_i}{\sum_j y_j} \cdot \log(\text{softmax}(s_i))
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
`lambda_weight`
</td>
<td>
(Optional) A lambdaweight to apply to the loss. Can be one
of <a href="../../../tfr/keras/losses/DCGLambdaWeight.md"><code>tfr.keras.losses.DCGLambdaWeight</code></a>,
<a href="../../../tfr/keras/losses/NDCGLambdaWeight.md"><code>tfr.keras.losses.NDCGLambdaWeight</code></a>, or,
<a href="../../../tfr/keras/losses/PrecisionLambdaWeight.md"><code>tfr.keras.losses.PrecisionLambdaWeight</code></a>.
</td>
</tr><tr>
<td>
`temperature`
</td>
<td>
(Optional) The temperature to use for scaling the logits.
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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L377-L384">View
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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L367-L375">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L424-L429">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true, y_pred, sample_weight=None
)
</code></pre>

See _RankingLoss.
