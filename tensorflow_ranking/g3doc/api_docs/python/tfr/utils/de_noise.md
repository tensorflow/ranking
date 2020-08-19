<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.de_noise" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.de_noise

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns a float `Tensor` as the de-noised `counts`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.de_noise(
    counts, noise, ratio=0.9
)
</code></pre>

<!-- Placeholder for "Used in" -->

The implementation is based on the the paper by Zhang and Xu: "Fast Exact
Maximum Likelihood Estimation for Mixture of Language Models." It assumes that
the observed `counts` are generated from a mixture of `noise` and the true
distribution: `ratio * noise_distribution + (1 - ratio) * true_distribution`,
where the contribution of `noise` is controlled by `ratio`. This method returns
the true distribution.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`counts`
</td>
<td>
A 2-D `Tensor` representing the observations. All values should be
nonnegative.
</td>
</tr><tr>
<td>
`noise`
</td>
<td>
A 2-D `Tensor` representing the noise distribution. This should be
the same shape as `counts`. All values should be positive and are
normalized to a simplex per row.
</td>
</tr><tr>
<td>
`ratio`
</td>
<td>
A float in (0, 1) representing the contribution from noise.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 2-D float `Tensor` and each row is a simplex.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `ratio` is not in (0,1).
</td>
</tr><tr>
<td>
`InvalidArgumentError`
</td>
<td>
if any of `counts` is negative or any of `noise` is
not positive.
</td>
</tr>
</table>
