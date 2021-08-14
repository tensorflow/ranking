description: LambdaWeight for ListMLE cost function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.ListMLELambdaWeight" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="individual_weights"/>
<meta itemprop="property" content="pair_weights"/>
</div>

# tfr.keras.losses.ListMLELambdaWeight

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L146-L154">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

LambdaWeight for ListMLE cost function.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.losses.ListMLELambdaWeight(
    rank_discount_fn, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`rank_discount_fn`
</td>
<td>
(function) The rank discount function.
</td>
</tr>
</table>

## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py#L151-L154">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

<h3 id="individual_weights"><code>individual_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L357-L363">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>individual_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.

<h3 id="pair_weights"><code>pair_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses_impl.py#L353-L355">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pair_weights(
    labels, ranks
)
</code></pre>

See `_LambdaWeight`.
