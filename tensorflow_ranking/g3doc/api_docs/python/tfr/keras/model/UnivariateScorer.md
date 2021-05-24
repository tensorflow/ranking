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
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L700-L764">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for univariate scorer.

Inherits From: [`Scorer`](../../../tfr/keras/model/Scorer.md)

<!-- Placeholder for "Used in" -->

The `UnivariateScorer` class is an abstract class to implement `score` in
`ModelBuilder` in tfr.keras with a univariate scoring function.

To be implemented by subclasses:

*   `_score_flattened()`: Contains the logic to do the univariate scoring on
    flattened context and example features.

Example subclass implementation:

```python
class SimpleUnivariateScorer(UnivariateScorer):

  def _score_flattened(self, context_features, example_features):
    x = tf.concat([tensor for tensor in example_features.values()], -1)
    return tf.keras.layers.Dense(1)(x)
```

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L742-L764">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[tf.Tensor, TensorDict]
</code></pre>

See `Scorer`.
