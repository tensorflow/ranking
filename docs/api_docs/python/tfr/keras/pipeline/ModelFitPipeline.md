description: Pipeline using model.fit to train a ranking tf.keras.Model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.ModelFitPipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_callbacks"/>
<meta itemprop="property" content="build_loss"/>
<meta itemprop="property" content="build_metrics"/>
<meta itemprop="property" content="build_weighted_metrics"/>
<meta itemprop="property" content="export_saved_model"/>
<meta itemprop="property" content="train_and_validate"/>
</div>

# tfr.keras.pipeline.ModelFitPipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L341-L608">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pipeline using `model.fit` to train a ranking `tf.keras.Model`.

Inherits From:
[`AbstractPipeline`](../../../tfr/keras/pipeline/AbstractPipeline.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.ModelFitPipeline(
    model_builder: <a href="../../../tfr/keras/model/AbstractModelBuilder.md"><code>tfr.keras.model.AbstractModelBuilder</code></a>,
    dataset_builder: <a href="../../../tfr/keras/pipeline/AbstractDatasetBuilder.md"><code>tfr.keras.pipeline.AbstractDatasetBuilder</code></a>,
    hparams: <a href="../../../tfr/keras/pipeline/PipelineHparams.md"><code>tfr.keras.pipeline.PipelineHparams</code></a>
)
</code></pre>

<!-- Placeholder for "Used in" -->

The `ModelFitPipeline` class is an abstract class inherit from
`AbstractPipeline` to train and validate a ranking `model` with `model.fit` in a
distributed strategy specified in hparams.

To be implemented by subclasses:

*   `build_loss()`: Contains the logic to build a `tf.keras.losses.Loss` or a
    dict or list of `tf.keras.losses.Loss`s to be optimized in training.
*   `build_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s to monitor and evaluate the training.
*   `build_weighted_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s which will take the weights.

Example subclass implementation:

```python
class BasicModelFitPipeline(ModelFitPipeline):

  def build_loss(self):
    return tfr.keras.losses.get('softmax_loss')

  def build_metrics(self):
    return [
        tfr.keras.metrics.get(
            'ndcg', topn=topn, name='ndcg_{}'.format(topn)
        ) for topn in [1, 5, 10]
    ]

  def build_weighted_metrics(self):
    return [
        tfr.keras.metrics.get(
            'ndcg', topn=topn, name='weighted_ndcg_{}'.format(topn)
        ) for topn in [1, 5, 10]
    ]
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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L77-L91">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_loss() -> Any
</code></pre>

Returns the loss for model.compile.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
loss = pipeline.build_loss()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.keras.losses.Loss` or a dict or list of `tf.keras.losses.Loss`.
</td>
</tr>

</table>

<h3 id="build_metrics"><code>build_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L93-L107">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_metrics() -> Any
</code></pre>

Returns a list of ranking metrics for `model.compile()`.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
metrics = pipeline.build_metrics()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list or a dict of `tf.keras.metrics.Metric`s.
</td>
</tr>

</table>

<h3 id="build_weighted_metrics"><code>build_weighted_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L109-L123">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_weighted_metrics() -> Any
</code></pre>

Returns a list of weighted ranking metrics for model.compile.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
weighted_metrics = pipeline.build_weighted_metrics()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list or a dict of `tf.keras.metrics.Metric`s.
</td>
</tr>

</table>

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
