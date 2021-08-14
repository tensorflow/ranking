description: Builds datasets from feature specs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.BaseDatasetBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_signatures"/>
<meta itemprop="property" content="build_train_dataset"/>
<meta itemprop="property" content="build_valid_dataset"/>
</div>

# tfr.keras.pipeline.BaseDatasetBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L824-L969">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds datasets from feature specs.

Inherits From:
[`AbstractDatasetBuilder`](../../../tfr/keras/pipeline/AbstractDatasetBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.BaseDatasetBuilder(
    context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]],
    example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]],
    training_only_example_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]],
    mask_feature_name: str,
    hparams: <a href="../../../tfr/keras/pipeline/DatasetHparams.md"><code>tfr.keras.pipeline.DatasetHparams</code></a>,
    training_only_context_spec: Optional[Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]]] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The `BaseDatasetBuilder` class is an abstract class inherit from
`AbstractDatasetBuilder` to serve training and validation datasets and
signatures for training `ModelFitPipeline`.

To be implemented by subclasses:

*   `_features_and_labels()`: Contains the logic to map a dict of tensors of
    dataset to feature tensors and label tensors.

Example subclass implementation:

```python
class SimpleDatasetBuilder(BaseDatasetBuilder):

  def _features_and_labels(self, features):
    label = features.pop("utility")
    return features, label
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`context_feature_spec`
</td>
<td>
Maps context (aka, query) names to feature specs.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
Maps example (aka, document) names to feature specs.
</td>
</tr><tr>
<td>
`training_only_example_spec`
</td>
<td>
Feature specs used for training only like
labels and per-example weights.
</td>
</tr><tr>
<td>
`mask_feature_name`
</td>
<td>
If set, populates the feature dictionary with this name
and the coresponding value is a `tf.bool` Tensor of shape [batch_size,
list_size] indicating the actual example is padded or not.
</td>
</tr><tr>
<td>
`hparams`
</td>
<td>
A dict containing model hyperparameters.
</td>
</tr><tr>
<td>
`training_only_context_spec`
</td>
<td>
Feature specs used for training only per-list
weights.
</td>
</tr>
</table>

## Methods

<h3 id="build_signatures"><code>build_signatures</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L963-L969">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_signatures(
    model: tf.keras.Model
) -> Any
</code></pre>

See `AbstractDatasetBuilder`.

<h3 id="build_train_dataset"><code>build_train_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L946-L952">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_train_dataset() -> tf.data.Dataset
</code></pre>

See `AbstractDatasetBuilder`.

<h3 id="build_valid_dataset"><code>build_valid_dataset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L954-L961">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_valid_dataset() -> tf.data.Dataset
</code></pre>

See `AbstractDatasetBuilder`.
