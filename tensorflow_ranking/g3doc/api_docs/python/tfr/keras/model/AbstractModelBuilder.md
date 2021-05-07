description: Interface to build a ranking tf.keras.Model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.AbstractModelBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="create_inputs"/>
<meta itemprop="property" content="preprocess"/>
<meta itemprop="property" content="score"/>
</div>

# tfr.keras.model.AbstractModelBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L67-L156">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface to build a ranking tf.keras.Model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.AbstractModelBuilder(
    mask_feature_name: str,
    name: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mask_feature_name`
</td>
<td>
name of 2D mask boolean feature.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
name of the Model.
</td>
</tr>
</table>

## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L139-L156">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build() -> tf.keras.Model
</code></pre>

Builds a Keras Model for Ranking Pipeline.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tf.keras.Model.
</td>
</tr>

</table>

<h3 id="create_inputs"><code>create_inputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L82-L92">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>create_inputs() -> Tuple[TensorDict, TensorDict, tf.Tensor]
</code></pre>

Creates context and example inputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of
</td>
</tr>
<tr>
<td>
`context_inputs`
</td>
<td>
maps from context feature keys to Keras Input.
</td>
</tr><tr>
<td>
`example_inputs`
</td>
<td>
maps from example feature keys to Keras Input.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
Keras Input for the mask feature.
</td>
</tr>
</table>

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L94-L115">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>preprocess(
    context_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Tuple[TensorDict, TensorDict]
</code></pre>

Preprocesses context and example inputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_inputs`
</td>
<td>
maps context feature keys to tf.keras.Input.
</td>
</tr><tr>
<td>
`example_inputs`
</td>
<td>
maps example feature keys to tf.keras.Input.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
[batch_size, list_size]-tensor of mask for valid examples.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of
</td>
</tr>
<tr>
<td>
`context_features`
</td>
<td>
maps from context feature keys to [batch_size,
feature_dims]-tensors of preprocessed context features.
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
maps from example feature keys to [batch_size,
list_size, feature_dims]-tensors of preprocessed example features.
</td>
</tr>
</table>

<h3 id="score"><code>score</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L117-L137">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>score(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[TensorLike, TensorDict]
</code></pre>

Scores all examples and returns outputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_features`
</td>
<td>
maps from context feature keys to [batch_size,
feature_dims]-tensors of preprocessed context features.
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
maps from example feature keys to [batch_size,
list_size, feature_dims]-tensors of preprocessed example features.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
[batch_size, list_size]-tensor of mask for valid examples.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A [batch_size, list_size]-tensor of logits or a dict mapping task name to
logits in the multi-task setting.
</td>
</tr>

</table>
