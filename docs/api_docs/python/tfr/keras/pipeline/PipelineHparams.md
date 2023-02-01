description: Hyperparameters used in ModelFitPipeline.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.PipelineHparams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="automatic_reduce_lr"/>
<meta itemprop="property" content="best_exporter_metric"/>
<meta itemprop="property" content="best_exporter_metric_higher_better"/>
<meta itemprop="property" content="cluster_resolver"/>
<meta itemprop="property" content="early_stopping_min_delta"/>
<meta itemprop="property" content="early_stopping_patience"/>
<meta itemprop="property" content="export_best_model"/>
<meta itemprop="property" content="loss_reduction"/>
<meta itemprop="property" content="loss_weights"/>
<meta itemprop="property" content="optimizer"/>
<meta itemprop="property" content="steps_per_execution"/>
<meta itemprop="property" content="strategy"/>
<meta itemprop="property" content="tpu"/>
<meta itemprop="property" content="use_weighted_metrics"/>
<meta itemprop="property" content="variable_partitioner"/>
</div>

# tfr.keras.pipeline.PipelineHparams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L247-L320">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Hyperparameters used in `ModelFitPipeline`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.PipelineHparams(
    model_dir: str,
    num_epochs: int,
    steps_per_epoch: int,
    validation_steps: int,
    learning_rate: float,
    loss: Union[str, Dict[str, str]],
    loss_reduction: str = tf.losses.Reduction.AUTO,
    optimizer: str = &#x27;adam&#x27;,
    loss_weights: Optional[Union[float, Dict[str, float]]] = None,
    steps_per_execution: int = 10,
    automatic_reduce_lr: bool = False,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    use_weighted_metrics: bool = False,
    export_best_model: bool = False,
    best_exporter_metric_higher_better: bool = False,
    best_exporter_metric: str = &#x27;loss&#x27;,
    strategy: Optional[str] = None,
    cluster_resolver: Optional[tf.distribute.cluster_resolver.ClusterResolver] = None,
    variable_partitioner: Optional[tf.distribute.experimental.partitioners.Partitioner] = None,
    tpu: Optional[str] = &#x27;&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->

Hyperparameters to be specified for ranking pipeline.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
A path to output the model and training data.
</td>
</tr><tr>
<td>
`num_epochs`<a id="num_epochs"></a>
</td>
<td>
An integer to specify the number of epochs of training.
</td>
</tr><tr>
<td>
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
An integer to specify the number of steps per epoch. When
it is None, going over the training data once is counted as an epoch.
</td>
</tr><tr>
<td>
`validation_steps`<a id="validation_steps"></a>
</td>
<td>
An integer to specify the number of validation steps in
each epoch. Note that a mini-batch of data will be evaluated in each step
and this is the number of steps taken for validation in each epoch.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
A float to indicate the learning rate of the optimizer.
</td>
</tr><tr>
<td>
`loss`<a id="loss"></a>
</td>
<td>
A string or a map to strings that indicate the loss to be used. When
`loss` is a string, all outputs and labels will be trained with the same
loss. When `loss` is a map, outputs and labels will be trained with losses
implied by the corresponding keys.
</td>
</tr><tr>
<td>
`loss_reduction`<a id="loss_reduction"></a>
</td>
<td>
An option in `tf.keras.losses.Reduction` to specify the
reduction method.
</td>
</tr><tr>
<td>
`optimizer`<a id="optimizer"></a>
</td>
<td>
An option in `tf.keras.optimizers` identifiers to specify the
optimizer to be used.
</td>
</tr><tr>
<td>
`loss_weights`<a id="loss_weights"></a>
</td>
<td>
None or a float or a map to floats that indicate the relative
weights for each loss. When not specified, all losses are applied with the
same weight 1.
</td>
</tr><tr>
<td>
`steps_per_execution`<a id="steps_per_execution"></a>
</td>
<td>
An integer to specify the number of steps executed in
each operation. Tuning this to optimize the training performance in
distributed training.
</td>
</tr><tr>
<td>
`automatic_reduce_lr`<a id="automatic_reduce_lr"></a>
</td>
<td>
A boolean to indicate whether to use
`ReduceLROnPlateau` callback.
</td>
</tr><tr>
<td>
`early_stopping_patience`<a id="early_stopping_patience"></a>
</td>
<td>
Number of epochs with no improvement after which
training will be stopped.
</td>
</tr><tr>
<td>
`early_stopping_min_delta`<a id="early_stopping_min_delta"></a>
</td>
<td>
Minimum change in the monitored quantity to
qualify as an improvement, i.e. an absolute change of less than
early_stopping_min_delta, will count as no improvement.
</td>
</tr><tr>
<td>
`use_weighted_metrics`<a id="use_weighted_metrics"></a>
</td>
<td>
A boolean to indicate whether to use weighted metrics.
</td>
</tr><tr>
<td>
`export_best_model`<a id="export_best_model"></a>
</td>
<td>
A boolean to indicate whether to export the best model
evaluated by the `best_exporter_metric` on the validation data.
</td>
</tr><tr>
<td>
`best_exporter_metric_higher_better`<a id="best_exporter_metric_higher_better"></a>
</td>
<td>
A boolean to indicate whether the
`best_exporter_metric` is the higher the better.
</td>
</tr><tr>
<td>
`best_exporter_metric`<a id="best_exporter_metric"></a>
</td>
<td>
A string to specify the metric used to monitor the
training and to export the best model. Default to the 'loss'.
</td>
</tr><tr>
<td>
`strategy`<a id="strategy"></a>
</td>
<td>
An option of strategies supported in `strategy_utils`. Choose from
["MirroredStrategy", "MultiWorkerMirroredStrategy",
"ParameterServerStrategy", "TPUStrategy"].
</td>
</tr><tr>
<td>
`cluster_resolver`<a id="cluster_resolver"></a>
</td>
<td>
A cluster_resolver to build strategy.
</td>
</tr><tr>
<td>
`variable_partitioner`<a id="variable_partitioner"></a>
</td>
<td>
Variable partitioner to be used in
ParameterServerStrategy.
</td>
</tr><tr>
<td>
`tpu`<a id="tpu"></a>
</td>
<td>
TPU address for TPUStrategy. Not used for other strategy.
</td>
</tr>
</table>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
automatic_reduce_lr<a id="automatic_reduce_lr"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
best_exporter_metric<a id="best_exporter_metric"></a>
</td>
<td>
`'loss'`
</td>
</tr><tr>
<td>
best_exporter_metric_higher_better<a id="best_exporter_metric_higher_better"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
cluster_resolver<a id="cluster_resolver"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
early_stopping_min_delta<a id="early_stopping_min_delta"></a>
</td>
<td>
`0.0`
</td>
</tr><tr>
<td>
early_stopping_patience<a id="early_stopping_patience"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
export_best_model<a id="export_best_model"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
loss_reduction<a id="loss_reduction"></a>
</td>
<td>
`'auto'`
</td>
</tr><tr>
<td>
loss_weights<a id="loss_weights"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
optimizer<a id="optimizer"></a>
</td>
<td>
`'adam'`
</td>
</tr><tr>
<td>
steps_per_execution<a id="steps_per_execution"></a>
</td>
<td>
`10`
</td>
</tr><tr>
<td>
strategy<a id="strategy"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
tpu<a id="tpu"></a>
</td>
<td>
`''`
</td>
</tr><tr>
<td>
use_weighted_metrics<a id="use_weighted_metrics"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
variable_partitioner<a id="variable_partitioner"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
