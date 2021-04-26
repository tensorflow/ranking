description: An no-op wrapper of datasets and signatures.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.NullDatasetBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_signatures"/>
<meta itemprop="property" content="build_train_dataset"/>
<meta itemprop="property" content="build_valid_dataset"/>
</div>

# tfr.keras.pipeline.NullDatasetBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L361-L379">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

An no-op wrapper of datasets and signatures.

Inherits From:
[`AbstractDatasetBuilder`](../../../tfr/keras/pipeline/AbstractDatasetBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.NullDatasetBuilder(
    train_dataset, valid_dataset, signatures=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="build_signatures"><code>build_signatures</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L377-L379">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_signatures(
    *arg, **kwargs
) -> Any
</code></pre>

See `AbstractDatasetBuilder`.

<h3 id="build_train_dataset"><code>build_train_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L369-L371">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_train_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

See `AbstractDatasetBuilder`.

<h3 id="build_valid_dataset"><code>build_valid_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L373-L375">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_valid_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

See `AbstractDatasetBuilder`.
