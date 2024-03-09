description: Computes the Rank Distil loss between y_true and y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.CoupledRankDistilLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.CoupledRankDistilLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L1645-L1734">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the Rank Distil loss between `y_true` and `y_pred`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.CoupledRankDistilLoss(
    reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
    name: Optional[str] = None,
    ragged: bool = False,
    sample_size: int = 8,
    topk: Optional[int] = None,
    temperature: Optional[float] = 1.0
)
</code></pre>

<!-- Placeholder for "Used in" -->

The Coupled-RankDistil loss ([Reddi et al, 2021][reddi2021]) is the
cross-entropy between k-Plackett's probability of logits (student) and labels
(teacher).

#### Standalone usage:

```
>>> tf.random.set_seed(1)
>>> y_true = [[0., 2., 1.], [1., 0., 2.]]
>>> ln = tf.math.log
>>> y_pred = [[0., ln(3.), ln(2.)], [0., ln(2.), ln(3.)]]
>>> loss = tfr.keras.losses.CoupledRankDistilLoss(topk=2, sample_size=1)
>>> loss(y_true, y_pred).numpy()
2.138333
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd',
              loss=tfr.keras.losses.CoupledRankDistilLoss())
```

#### Definition:

The k-Plackett's probability model is defined as: $$ \mathcal{P}_k(\pi|s) =
\frac{1}{(N-k)!} \\ \frac{\prod_{i=1}^k exp(s_{\pi(i)})}{\sum_{j=k}^N
log(exp(s_{\pi(i)}))}. $$

The Coupled-RankDistil loss is defined as: $$ \mathcal{L}(y, s) = -\sum_{\pi}
\mathcal{P}_k(\pi|y) log\mathcal{P}(\pi|s) \\ = \mathcal{E}_{\pi \sim
\matcal{P}(.|y)} [-\log \mathcal{P}(\pi|s)] $$

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
- [RankDistil: Knowledge Distillation for Ranking, Reddi et al, 2021][reddi2021]
</td>
</tr>

</table>

[reddi2021]: https://research.google/pubs/pub50695/

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
`ragged`<a id="ragged"></a>
</td>
<td>
(Optional) If True, this loss will accept ragged tensors. If
False, this loss will accept dense tensors.
</td>
</tr><tr>
<td>
`sample_size`<a id="sample_size"></a>
</td>
<td>
(Optional) Number of permutations to sample from teacher
scores. Defaults to 8.
</td>
</tr><tr>
<td>
`topk`<a id="topk"></a>
</td>
<td>
(Optional) top-k entries over which order is matched. A penalty is
applied over non top-k items. Defaults to `None`, which treats top-k as
all entries in the list.
</td>
</tr><tr>
<td>
`temperature`<a id="temperature"></a>
</td>
<td>
(Optional) A float number to modify the logits as
`logits=logits/temperature`. Defaults to 1.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L1727-L1734">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L250-L258">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    y_pred: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    sample_weight: Optional[utils.TensorLike] = None
) -> tf.Tensor
</code></pre>

See tf.keras.losses.Loss.
