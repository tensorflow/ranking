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
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L87-L109">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Keras serializable class for DCG.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.DCGLambdaWeight(
    topn=None, gain_fn=None, rank_discount_fn=None, normalized=False,
    smooth_fraction=0.0, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`topn`
</td>
<td>
(int) The topn for the DCG metric.
</td>
</tr><tr>
<td>
`gain_fn`
</td>
<td>
(function) Transforms labels.
</td>
</tr><tr>
<td>
`rank_discount_fn`
</td>
<td>
(function) The rank discount function.
</td>
</tr><tr>
<td>
`normalized`
</td>
<td>
(bool) If True, normalize weight by the max DCG.
</td>
</tr><tr>
<td>
`smooth_fraction`
</td>
<td>
(float) parameter to control the contribution from
LambdaMART.
</td>
</tr>
</table>

## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L102-L109">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

<h3 id="individual_weights"><code>individual_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L275-L290">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>individual_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.

<h3 id="pair_weights"><code>pair_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L215-L273">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pair_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.
