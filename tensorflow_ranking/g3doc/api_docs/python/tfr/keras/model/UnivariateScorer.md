description: Interface for univariate scorer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.UnivariateScorer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
</div>

# tfr.keras.model.UnivariateScorer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L414-L448">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for univariate scorer.

Inherits From: [`Scorer`](../../../tfr/keras/model/Scorer.md)

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L426-L448">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[tf.Tensor, TensorDict]
</code></pre>

Scores all examples and returns logits.
