description: A no-op wrapper of datasets and signatures.

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
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L785-L821">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A no-op wrapper of datasets and signatures.

Inherits From:
[`AbstractDatasetBuilder`](../../../tfr/keras/pipeline/AbstractDatasetBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.NullDatasetBuilder(
    train_dataset, valid_dataset, signatures=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example usage:

```python
train_dataset = tf.data.Dataset(...)
valid_dataset = tf.data.Dataset(...)
dataset_builder = NullDatasetBuilder(train_dataset, valid_dataset)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`train_dataset`
</td>
<td>
A `tf.data.Dataset` for training.
</td>
</tr><tr>
<td>
`valid_dataset`
</td>
<td>
A `tf.data.Dataset` for validation.
</td>
</tr><tr>
<td>
`signatures`
</td>
<td>
A dict of signatures that formulate the model in functions
that render the input data with given types. When None, no signatures
assigned.
</td>
</tr>
</table>

## Methods

<h3 id="build_signatures"><code>build_signatures</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L819-L821">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_signatures(
    *arg, **kwargs
) -> Any
</code></pre>

See `AbstractDatasetBuilder`.

<h3 id="build_train_dataset"><code>build_train_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L811-L813">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_train_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

See `AbstractDatasetBuilder`.

<h3 id="build_valid_dataset"><code>build_valid_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L815-L817">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_valid_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

See `AbstractDatasetBuilder`.
