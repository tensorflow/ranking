description: Hparams used in pipeline.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.PipelineHparams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="automatic_reduce_lr"/>
<meta itemprop="property" content="best_exporter_metric"/>
<meta itemprop="property" content="best_exporter_metric_higher_better"/>
<meta itemprop="property" content="export_best_model"/>
<meta itemprop="property" content="loss_reduction"/>
<meta itemprop="property" content="loss_weights"/>
<meta itemprop="property" content="master"/>
<meta itemprop="property" content="optimizer"/>
<meta itemprop="property" content="steps_per_execution"/>
<meta itemprop="property" content="strategy"/>
<meta itemprop="property" content="use_weighted_metrics"/>
</div>

# tfr.keras.pipeline.PipelineHparams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L63-L81">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Hparams used in pipeline.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.pipeline.PipelineHparams(
    num_epochs: int,
    num_train_steps: int,
    num_valid_steps: int,
    learning_rate: float,
    loss: Union[str, Dict[str, str]],
    loss_reduction: str = &#x27;auto&#x27;,
    optimizer: str = &#x27;adam&#x27;,
    loss_weights: Optional[Union[float, Dict[str, float]]] = None,
    steps_per_execution: int = 10,
    automatic_reduce_lr: bool = False,
    use_weighted_metrics: bool = False,
    export_best_model: bool = False,
    best_exporter_metric_higher_better: bool = False,
    best_exporter_metric: str = &#x27;loss&#x27;,
    strategy: Optional[str] = None,
    master: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`model_dir`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`num_epochs`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`num_train_steps`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`num_valid_steps`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`learning_rate`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`loss`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`loss_reduction`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`optimizer`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`loss_weights`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`steps_per_execution`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`automatic_reduce_lr`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`use_weighted_metrics`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`export_best_model`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`best_exporter_metric_higher_better`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`best_exporter_metric`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`strategy`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`master`
</td>
<td>
Dataclass field
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
master<a id="master"></a>
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
use_weighted_metrics<a id="use_weighted_metrics"></a>
</td>
<td>
`False`
</td>
</tr>
</table>
