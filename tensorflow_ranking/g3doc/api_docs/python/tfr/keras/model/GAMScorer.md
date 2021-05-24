description: Univariate scorer using GAM.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.GAMScorer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfr.keras.model.GAMScorer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L807-L852">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Univariate scorer using GAM.

Inherits From:
[`UnivariateScorer`](../../../tfr/keras/model/UnivariateScorer.md),
[`Scorer`](../../../tfr/keras/model/Scorer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.GAMScorer(
    **gam_kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The scorer implements Neural Generalized Additive Ranking Model, which is an
additive ranking model. See the [paper](https://arxiv.org/abs/2005.02553) for
more details.

#### Example usage:

```python
scorer=GAMScorer(hidden_layer_dims=[16])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`**gam_kwargs`
</td>
<td>
A dict of keyward arguments for GAM layers. Please see
`tfr.keras.layers.GAMlayer` for specific list of keyword arguments.
</td>
</tr>
</table>

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
