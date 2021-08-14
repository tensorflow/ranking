description: Pipeline for multi-task training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.MultiTaskPipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_callbacks"/>
<meta itemprop="property" content="build_loss"/>
<meta itemprop="property" content="build_metrics"/>
<meta itemprop="property" content="build_weighted_metrics"/>
<meta itemprop="property" content="export_saved_model"/>
<meta itemprop="property" content="train_and_validate"/>
</div>

# tfr.keras.pipeline.MultiTaskPipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L691-L782">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pipeline for multi-task training.

Inherits From:
[`ModelFitPipeline`](../../../tfr/keras/pipeline/ModelFitPipeline.md),
[`AbstractPipeline`](../../../tfr/keras/pipeline/AbstractPipeline.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.MultiTaskPipeline(
    model_builder: <a href="../../../tfr/keras/model/AbstractModelBuilder.md"><code>tfr.keras.model.AbstractModelBuilder</code></a>,
    dataset_builder: <a href="../../../tfr/keras/pipeline/AbstractDatasetBuilder.md"><code>tfr.keras.pipeline.AbstractDatasetBuilder</code></a>,
    hparams: <a href="../../../tfr/keras/pipeline/PipelineHparams.md"><code>tfr.keras.pipeline.PipelineHparams</code></a>
)
</code></pre>

<!-- Placeholder for "Used in" -->

This handles a set of losses and labels. It is intended to mainly work with
`MultiLabelDatasetBuilder`.

Use subclassing to customize the losses and metrics.

#### Example usage:

```python
context_feature_spec = {}
example_feature_spec = {
    "example_feature_1": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=0.0)
}
mask_feature_name = "list_mask"
label_spec_tuple = ("utility",
                    tf.io.FixedLenFeature(
                        shape=(1,),
                        dtype=tf.float32,
                        default_value=_PADDING_LABEL))
label_spec = {"task1": label_spec_tuple, "task2": label_spec_tuple}
weight_spec = ("weight",
               tf.io.FixedLenFeature(
                   shape=(1,), dtype=tf.float32, default_value=1.))
dataset_hparams = DatasetHparams(
    train_input_pattern="train.dat",
    valid_input_pattern="valid.dat",
    train_batch_size=128,
    valid_batch_size=128)
pipeline_hparams = PipelineHparams(
    model_dir="model/",
    num_epochs=2,
    steps_per_epoch=5,
    validation_steps=2,
    learning_rate=0.01,
    loss={
        "task1": "softmax_loss",
        "task2": "pairwise_logistic_loss"
    },
    loss_weights={
        "task1": 1.0,
        "task2": 2.0
    },
    export_best_model=True)
model_builder = MultiTaskModelBuilder(...)
dataset_builder = MultiLabelDatasetBuilder(
    context_feature_spec,
    example_feature_spec,
    mask_feature_name,
    label_spec,
    dataset_hparams,
    sample_weight_spec=weight_spec)
pipeline = MultiTaskPipeline(model_builder, dataset_builder, pipeline_hparams)
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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L750-L759">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_loss() -> Dict[str, tf.keras.losses.Loss]
</code></pre>

See `AbstractPipeline`.

<h3 id="build_metrics"><code>build_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L761-L770">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_metrics() -> Dict[str, List[tf.keras.metrics.Metric]]
</code></pre>

See `AbstractPipeline`.

<h3 id="build_weighted_metrics"><code>build_weighted_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L772-L782">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_weighted_metrics() -> Dict[str, List[tf.keras.metrics.Metric]]
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
