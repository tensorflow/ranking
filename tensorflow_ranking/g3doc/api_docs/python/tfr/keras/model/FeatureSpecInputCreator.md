description: InputCreator with feature specs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.FeatureSpecInputCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfr.keras.model.FeatureSpecInputCreator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L225-L275">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

InputCreator with feature specs.

Inherits From: [`InputCreator`](../../../tfr/keras/model/InputCreator.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.FeatureSpecInputCreator(
    context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]],
    example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]]
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L240-L275">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__() -> Tuple[TensorDict, TensorDict]
</code></pre>

See `InputCreator`.
