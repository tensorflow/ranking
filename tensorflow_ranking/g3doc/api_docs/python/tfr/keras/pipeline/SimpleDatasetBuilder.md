description: Builds datasets from feature specs with a single label spec.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.SimpleDatasetBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_signatures"/>
<meta itemprop="property" content="build_train_dataset"/>
<meta itemprop="property" content="build_valid_dataset"/>
</div>

# tfr.keras.pipeline.SimpleDatasetBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L988-L1079">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds datasets from feature specs with a single label spec.

Inherits From:
[`BaseDatasetBuilder`](../../../tfr/keras/pipeline/BaseDatasetBuilder.md),
[`AbstractDatasetBuilder`](../../../tfr/keras/pipeline/AbstractDatasetBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.SimpleDatasetBuilder(
    context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]],
    example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.
        RaggedFeature]],
    mask_feature_name: str,
    label_spec: Tuple[str, tf.io.FixedLenFeature],
    hparams: <a href="../../../tfr/keras/pipeline/DatasetHparams.md"><code>tfr.keras.pipeline.DatasetHparams</code></a>,
    sample_weight_spec: Optional[Tuple[str, tf.io.FixedLenFeature]] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This supports a single dataset with a single label, which is supposed to be a
dense Tensor.

#### Example usage:

```python
context_feature_spec = {}
example_feature_spec = {
    "example_feature_1": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=0.0)
}
mask_feature_name = "list_mask"
label_spec = {
    "utility": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=0.0)
}
dataset_hparams = DatasetHparams(
    train_input_pattern="train.dat",
    valid_input_pattern="valid.dat",
    train_batch_size=128,
    valid_batch_size=128)
dataset_builder = SimpleDatasetBuilder(
    context_feature_spec,
    example_feature_spec,
    mask_feature_name,
    label_spec,
    dataset_hparams)
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
`mask_feature_name`
</td>
<td>
If set, populates the feature dictionary with this name
and the coresponding value is a `tf.bool` Tensor of shape [batch_size,
list_size] indicating the actual example is padded or not.
</td>
</tr><tr>
<td>
`label_spec`
</td>
<td>
A tuple of the label name and a tf.io.FixedLenFeature spec, or
a dict that maps task name to label spec in multi-task setting.
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
`sample_weight_spec`
</td>
<td>
Feature spec for per-example weight.
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
