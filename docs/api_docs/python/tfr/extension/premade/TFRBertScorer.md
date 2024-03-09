description: Univariate BERT-based scorer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.premade.TFRBertScorer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfr.extension.premade.TFRBertScorer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L96-L118">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Univariate BERT-based scorer.

Inherits From:
[`UnivariateScorer`](../../../tfr/keras/model/UnivariateScorer.md),
[`Scorer`](../../../tfr/keras/model/Scorer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfr.extension.premade.tfrbert_task.TFRBertScorer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.extension.premade.TFRBertScorer(
    encoder: tf.keras.Model,
    bert_output_dropout: float,
    name: str = &#x27;tfrbert&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L741-L763">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    context_features: <a href="../../../tfr/keras/model/TensorDict.md"><code>tfr.keras.model.TensorDict</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorDict.md"><code>tfr.keras.model.TensorDict</code></a>,
    mask: tf.Tensor
) -> Union[tf.Tensor, <a href="../../../tfr/keras/model/TensorDict.md"><code>tfr.keras.model.TensorDict</code></a>]
</code></pre>

See `Scorer`.
