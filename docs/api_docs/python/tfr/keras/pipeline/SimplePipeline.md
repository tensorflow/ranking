description: Pipleine for single-task training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.SimplePipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_callbacks"/>
<meta itemprop="property" content="build_loss"/>
<meta itemprop="property" content="build_metrics"/>
<meta itemprop="property" content="build_weighted_metrics"/>
<meta itemprop="property" content="export_saved_model"/>
<meta itemprop="property" content="train_and_validate"/>
</div>

# tfr.keras.pipeline.SimplePipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L617-L688">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pipleine for single-task training.

Inherits From:
[`ModelFitPipeline`](../../../tfr/keras/pipeline/ModelFitPipeline.md),
[`AbstractPipeline`](../../../tfr/keras/pipeline/AbstractPipeline.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.SimplePipeline(
    model_builder: <a href="../../../tfr/keras/model/AbstractModelBuilder.md"><code>tfr.keras.model.AbstractModelBuilder</code></a>,
    dataset_builder: <a href="../../../tfr/keras/pipeline/AbstractDatasetBuilder.md"><code>tfr.keras.pipeline.AbstractDatasetBuilder</code></a>,
    hparams: <a href="../../../tfr/keras/pipeline/PipelineHparams.md"><code>tfr.keras.pipeline.PipelineHparams</code></a>
)
</code></pre>

<!-- Placeholder for "Used in" -->

This handles a single loss and works with `SimpleDatasetBuilder`. This can also
work with `MultiLabelDatasetBuilder`. In this case, the same loss, as well as
all metrics, will be applied to all labels and their predictions uniformly.

Use subclassing to customize the loss and metrics.

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
pipeline_hparams = pipeline.PipelineHparams(
    model_dir="model/",
    num_epochs=2,
    steps_per_epoch=5,
    validation_steps=2,
    learning_rate=0.01,
    loss="softmax_loss")
model_builder = SimpleModelBuilder(
    context_feature_spec, example_feature_spec, mask_feature_name)
dataset_builder = SimpleDatasetBuilder(
    context_feature_spec,
    example_feature_spec,
    mask_feature_name,
    label_spec,
    dataset_hparams)
pipeline = SimplePipeline(model_builder, dataset_builder, pipeline_hparams)
pipeline.train_and_validate(verbose=1)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_builder`
</td>
<td>
A `ModelBuilder` instance for model fit.
</td>
</tr><tr>
<td>
`dataset_builder`
</td>
<td>
An `AbstractDatasetBuilder` instance to load train and
validate datasets and create signatures for SavedModel.
</td>
</tr><tr>
<td>
`hparams`
</td>
<td>
A dict containing model hyperparameters.
</td>
</tr>
</table>

## Methods

<h3 id="build_callbacks"><code>build_callbacks</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L443-L490">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_callbacks() -> List[tf.keras.callbacks.Callback]
</code></pre>

Sets up Callbacks.

#### Example usage:

```python
model_builder = ModelBuilder(...)
dataset_builder = DatasetBuilder(...)
hparams = PipelineHparams(...)
pipeline = BasicModelFitPipeline(model_builder, dataset_builder, hparams)
callbacks = pipeline.build_callbacks()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of `tf.keras.callbacks.Callback` or a
`tf.keras.callbacks.CallbackList` for tensorboard and checkpoint.
</td>
</tr>

</table>

<h3 id="build_loss"><code>build_loss</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L665-L671">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_loss() -> tf.keras.losses.Loss
</code></pre>

See `AbstractPipeline`.

<h3 id="build_metrics"><code>build_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L673-L679">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_metrics() -> List[tf.keras.metrics.Metric]
</code></pre>

See `AbstractPipeline`.

<h3 id="build_weighted_metrics"><code>build_weighted_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L681-L688">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_weighted_metrics() -> List[tf.keras.metrics.Metric]
</code></pre>

See `AbstractPipeline`.

<h3 id="export_saved_model"><code>export_saved_model</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L492-L517">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_saved_model(
    model: tf.keras.Model,
    export_to: str,
    checkpoint: Optional[tf.train.Checkpoint] = None
)
</code></pre>

Exports the trained model with signatures.

#### Example usage:

```python
model_builder = ModelBuilder(...)
dataset_builder = DatasetBuilder(...)
hparams = PipelineHparams(...)
pipeline = BasicModelFitPipeline(model_builder, dataset_builder, hparams)
pipeline.export_saved_model(model_builder.build(), 'saved_model/')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
Model to be saved.
</td>
</tr><tr>
<td>
`export_to`
</td>
<td>
Specifies the directory the model is be exported to.
</td>
</tr><tr>
<td>
`checkpoint`
</td>
<td>
If given, export the model with weights from this checkpoint.
</td>
</tr>
</table>

<h3 id="train_and_validate"><code>train_and_validate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L519-L608">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train_and_validate(
    verbose=0
)
</code></pre>

Main function to train the model with TPU strategy.

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
pipeline_hparams = pipeline.PipelineHparams(
    model_dir="model/",
    num_epochs=2,
    steps_per_epoch=5,
    validation_steps=2,
    learning_rate=0.01,
    loss="softmax_loss")
model_builder = SimpleModelBuilder(
    context_feature_spec, example_feature_spec, mask_feature_name)
dataset_builder = SimpleDatasetBuilder(
    context_feature_spec,
    example_feature_spec,
    mask_feature_name,
    label_spec,
    dataset_hparams)
pipeline = BasicModelFitPipeline(
    model_builder, dataset_builder, pipeline_hparams)
pipeline.train_and_validate(verbose=1)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`verbose`
</td>
<td>
An int for the verbosity level.
</td>
</tr>
</table>
