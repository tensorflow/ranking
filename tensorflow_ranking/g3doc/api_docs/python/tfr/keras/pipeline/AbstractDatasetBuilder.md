description: Interface for datasets and signatures.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.AbstractDatasetBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="build_signatures"/>
<meta itemprop="property" content="build_train_dataset"/>
<meta itemprop="property" content="build_valid_dataset"/>
</div>

# tfr.keras.pipeline.AbstractDatasetBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L43-L59">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for datasets and signatures.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="build_signatures"><code>build_signatures</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L56-L59">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_signatures(
    *arg, **kwargs
) -> Any
</code></pre>

Returns the signatures to export saved model.

<h3 id="build_train_dataset"><code>build_train_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L46-L49">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_train_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

Returns the training dataset.

<h3 id="build_valid_dataset"><code>build_valid_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L51-L54">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_valid_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

Returns the validation dataset.
