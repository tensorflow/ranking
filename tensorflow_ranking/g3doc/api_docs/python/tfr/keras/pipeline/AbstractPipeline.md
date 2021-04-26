description: Defines the interface for a pipeline to train a ranking
tf.keras.Model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.AbstractPipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="build_loss"/>
<meta itemprop="property" content="build_metrics"/>
<meta itemprop="property" content="build_weighted_metrics"/>
<meta itemprop="property" content="train_and_validate"/>
</div>

# tfr.keras.pipeline.AbstractPipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L19-L40">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Defines the interface for a pipeline to train a ranking tf.keras.Model.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="build_loss"><code>build_loss</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L22-L25">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_loss() -> Any
</code></pre>

Returns the loss for model.compile.

<h3 id="build_metrics"><code>build_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L27-L30">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_metrics() -> Any
</code></pre>

Returns a list of ranking metrics for model.compile.

<h3 id="build_weighted_metrics"><code>build_weighted_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L32-L35">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_weighted_metrics() -> Any
</code></pre>

Returns a list of weighted ranking metrics for model.compile.

<h3 id="train_and_validate"><code>train_and_validate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L37-L40">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>train_and_validate(
    *arg, **kwargs
) -> Any
</code></pre>

Constructs and runs the training pipeline.
