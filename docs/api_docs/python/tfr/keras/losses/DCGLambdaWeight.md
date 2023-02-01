description: Keras serializable class for DCG.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.DCGLambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.keras.losses.DCGLambdaWeight

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L107-L130">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Keras serializable class for DCG.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.DCGLambdaWeight(
    topn: Optional[int] = None,
    gain_fn: Optional[<a href="../../../tfr/keras/utils/GainFunction.md"><code>tfr.keras.utils.GainFunction</code></a>] = None,
    rank_discount_fn: Optional[<a href="../../../tfr/keras/utils/GainFunction.md"><code>tfr.keras.utils.GainFunction</code></a>] = None,
    normalized: bool = False,
    smooth_fraction: float = 0.0,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`topn`<a id="topn"></a>
</td>
<td>
(int) The topn for the DCG metric.
</td>
</tr><tr>
<td>
`gain_fn`<a id="gain_fn"></a>
</td>
<td>
(function) Transforms labels.
</td>
</tr><tr>
<td>
`rank_discount_fn`<a id="rank_discount_fn"></a>
</td>
<td>
(function) The rank discount function.
</td>
</tr><tr>
<td>
`normalized`<a id="normalized"></a>
</td>
<td>
(bool) If True, normalize weight by the max DCG.
</td>
</tr><tr>
<td>
`smooth_fraction`<a id="smooth_fraction"></a>
</td>
<td>
(float) parameter to control the contribution from
LambdaMART.
</td>
</tr>
</table>

## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L123-L130">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

<h3 id="individual_weights"><code>individual_weights</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L267-L282">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>individual_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.

<h3 id="pair_weights"><code>pair_weights</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L241-L265">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pair_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.
