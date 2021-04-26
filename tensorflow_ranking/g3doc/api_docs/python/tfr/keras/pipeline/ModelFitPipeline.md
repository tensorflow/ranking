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
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L97-L271">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Pipeline using model.fit to train a ranking tf.keras.Model.

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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L162-L198">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_callbacks() -> List[tf.keras.callbacks.Callback]
</code></pre>

Sets up Callbacks.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of callbacks for tensorboard and checkpoint.
</td>
</tr>

</table>

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

<h3 id="export_saved_model"><code>export_saved_model</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L200-L215">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_saved_model(
    model: tf.keras.Model,
    export_to: str,
    checkpoint: Optional[tf.train.Checkpoint] = None
)
</code></pre>

Exports the trained model with signatures.

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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L217-L271">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train_and_validate(
    verbose=0
)
</code></pre>

Main function to train the model with TPU strategy.

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
