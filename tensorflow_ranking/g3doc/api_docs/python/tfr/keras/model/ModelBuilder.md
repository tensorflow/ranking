description: Builds a tf.keras.Model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.ModelBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="create_inputs"/>
<meta itemprop="property" content="preprocess"/>
<meta itemprop="property" content="score"/>
</div>

# tfr.keras.model.ModelBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L159-L214">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a tf.keras.Model.

Inherits From:
[`AbstractModelBuilder`](../../../tfr/keras/model/AbstractModelBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.ModelBuilder(
    input_creator: Callable[[], Tuple[TensorDict, TensorDict]],
    preprocessor: Callable[[TensorDict, TensorDict, tf.Tensor], Tuple[TensorDict, TensorDict]],
    scorer: Callable[[TensorDict, TensorDict, tf.Tensor], Union[TensorLike, TensorDict]],
    mask_feature_name: str,
    name: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This class implements the AbstractModelBuilder by delegating the class behaviors
to the following implementors that can be specified by callers:

*   input_creator: A callable or a class like `InputCreator` to implement
    `create_inputs`.
*   preprocessor: A callable or a class like `Preprocessor` to implement
    `preprocess`.
*   scorer: A callable or a class like `Scorer` to implement `score`.

Users can subclass those implementor classes and pass the objects into this
class to build a tf.keras.Model.

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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L191-L196">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_inputs() -> Tuple[TensorDict, TensorDict, tf.Tensor]
</code></pre>

See `AbstractModelBuilder`.

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L198-L205">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>preprocess(
    context_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Tuple[TensorDict, TensorDict]
</code></pre>

See `AbstractModelBuilder`.

<h3 id="score"><code>score</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L207-L214">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>score(
    context_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_features: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Union[TensorLike, TensorDict]
</code></pre>

See `AbstractModelBuilder`.
