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
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L146-L245">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for datasets and signatures.

<!-- Placeholder for "Used in" -->

The `AbstractDatasetBuilder` class is an abstract class to serve data in
tfr.keras. A `DatasetBuilder` will be passed to an instance of
`AbstractPipeline` and called to serve the training and validation datasets and
to define the serving signatures for saved models to treat the corresponding
format of data.

To be implemented by subclasses:

*   `build_train_dataset()`: Contains the logic to build a `tf.data.Dataset` for
    training.
*   `build_valid_dataset()`: Contains the logic to build a `tf.data.Dataset` for
    validation.
*   `build_signatures()`: Contains the logic to build a dict of signatures that
    formulate the model in functions that render the input data with given
    format.

Example subclass implementation:

```python
class NullDatasetBuilder(AbstractDatasetBuilder):

  def __init__(self, train_dataset, valid_dataset, signatures=None):
    self._train_dataset = train_dataset
    self._valid_dataset = valid_dataset
    self._signatures = signatures

  def build_train_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
    return self._train_dataset

  def build_valid_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
    return self._valid_dataset

  def build_signatures(self, *arg, **kwargs) -> Any:
    return self._signatures
```

## Methods

<h3 id="build_signatures"><code>build_signatures</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L227-L245">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_signatures(
    *arg, **kwargs
) -> Any
</code></pre>

Returns the signatures to export a SavedModel.

#### Example usage:

```python
dataset_builder = NullDatasetBuilder(train_data, valid_data)
signatures = dataset_builder.build_signatures()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*arg`
</td>
<td>
arguments that might be used to build signatures.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments that might be used to build signatures.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a dict of concrete functions.
</td>
</tr>

</table>

<h3 id="build_train_dataset"><code>build_train_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L186-L204">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_train_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

Returns the training dataset.

#### Example usage:

```python
dataset_builder = NullDatasetBuilder(train_data, valid_data)
train_dataset = dataset_builder.build_train_dataset()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*arg`
</td>
<td>
arguments that might be used to build training dataset.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments that might be used to build training dataset.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.data.Dataset`.
</td>
</tr>

</table>

<h3 id="build_valid_dataset"><code>build_valid_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L206-L225">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_valid_dataset(
    *arg, **kwargs
) -> tf.data.Dataset
</code></pre>

Returns the validation dataset.

#### Example usage:

```python
dataset_builder = NullDatasetBuilder(train_data, valid_data)
valid_dataset = dataset_builder.build_valid_dataset()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*arg`
</td>
<td>
arguments that might be used to build validation dataset.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments that might be used to build validation
dataset.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.data.Dataset`.
</td>
</tr>

</table>
