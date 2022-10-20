description: Task object for TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.task.RankingTask" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="aggregate_logs"/>
<meta itemprop="property" content="build_inputs"/>
<meta itemprop="property" content="build_losses"/>
<meta itemprop="property" content="build_metrics"/>
<meta itemprop="property" content="build_model"/>
<meta itemprop="property" content="create_optimizer"/>
<meta itemprop="property" content="inference_step"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="process_compiled_metrics"/>
<meta itemprop="property" content="process_metrics"/>
<meta itemprop="property" content="reduce_aggregated_logs"/>
<meta itemprop="property" content="train_step"/>
<meta itemprop="property" content="validation_step"/>
<meta itemprop="property" content="with_name_scope"/>
<meta itemprop="property" content="loss"/>
</div>

# tfr.extension.task.RankingTask

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L174-L263">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Task object for TF-Ranking.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.extension.task.RankingTask(
    params,
    model_builder: <a href="../../../tfr/keras/model/AbstractModelBuilder.md"><code>tfr.keras.model.AbstractModelBuilder</code></a>,
    context_feature_spec: <a href="../../../tfr/extension/task/FeatureSpec.md"><code>tfr.extension.task.FeatureSpec</code></a> = None,
    example_feature_spec: <a href="../../../tfr/extension/task/FeatureSpec.md"><code>tfr.extension.task.FeatureSpec</code></a> = None,
    label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
    dataset_fn: Optional[Callable[[], tf.data.Dataset]] = None,
    logging_dir: Optional[str] = None,
    name: Optional[str] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`params`<a id="params"></a>
</td>
<td>
the task configuration instance, which can be any of dataclass,
ConfigDict, namedtuple, etc.
</td>
</tr><tr>
<td>
`logging_dir`<a id="logging_dir"></a>
</td>
<td>
a string pointing to where the model, summaries etc. will be
saved. You can also write additional stuff in this directory.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
the task name.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `logging_dir`<a id="logging_dir"></a> </td> <td>

</td> </tr><tr> <td> `name`<a id="name"></a> </td> <td> Returns the name of this
module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes parent
module names. </td> </tr><tr> <td> `name_scope`<a id="name_scope"></a> </td>
<td> Returns a `tf.name_scope` instance for this class. </td> </tr><tr> <td>
`non_trainable_variables`<a id="non_trainable_variables"></a> </td> <td>
Sequence of non-trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`submodules`<a id="submodules"></a> </td> <td> Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```

</td> </tr><tr> <td> `task_config`<a id="task_config"></a> </td> <td>

</td> </tr><tr> <td> `trainable_variables`<a id="trainable_variables"></a> </td>
<td> Sequence of trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`variables`<a id="variables"></a> </td> <td> Sequence of variables owned by this
module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>

## Methods

<h3 id="aggregate_logs"><code>aggregate_logs</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>aggregate_logs(
    state, step_logs
)
</code></pre>

Optional aggregation over logs returned from a validation step.

Given step_logs from a validation step, this function aggregates the logs after
each eval_step() (see eval_reduce() function in official/core/base_trainer.py).
It runs on CPU and can be used to aggregate metrics during validation, when
there are too many metrics that cannot fit into TPU memory. Note that this may
increase latency due to data transfer between TPU and CPU. Also, the step output
from a validation step may be a tuple with elements from replicas, and a
concatenation of the elements is needed in such case.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`state`
</td>
<td>
The current state of training, for example, it can be a sequence of
metrics.
</td>
</tr><tr>
<td>
`step_logs`
</td>
<td>
Logs from a validation step. Can be a dictionary.
</td>
</tr>
</table>

<h3 id="build_inputs"><code>build_inputs</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L199-L206">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_inputs(
    params, input_context=None
)
</code></pre>

Returns a dataset or a nested structure of dataset functions.

Dataset functions define per-host datasets with the per-replica batch size. With
distributed training, this method runs on remote hosts.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`params`
</td>
<td>
hyperparams to create input pipelines, which can be any of
dataclass, ConfigDict, namedtuple, etc.
</td>
</tr><tr>
<td>
`input_context`
</td>
<td>
optional distribution input pipeline context.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A nested structure of per-replica input functions.
</td>
</tr>

</table>

<h3 id="build_losses"><code>build_losses</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L208-L215">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_losses(
    labels, model_outputs, aux_losses=None
) -> tf.Tensor
</code></pre>

Standard interface to compute losses.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`labels`
</td>
<td>
optional label tensors.
</td>
</tr><tr>
<td>
`model_outputs`
</td>
<td>
a nested structure of output tensors.
</td>
</tr><tr>
<td>
`aux_losses`
</td>
<td>
auxiliary loss tensors, i.e. `losses` in keras.Model.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The total loss tensor.
</td>
</tr>

</table>

<h3 id="build_metrics"><code>build_metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L217-L228">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_metrics(
    training=None
)
</code></pre>

Gets streaming metrics for training/validation.

<h3 id="build_model"><code>build_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L196-L197">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_model()
</code></pre>

[Optional] Creates model architecture.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A model instance.
</td>
</tr>

</table>

<h3 id="create_optimizer"><code>create_optimizer</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_optimizer(
    optimizer_config: OptimizationConfig,
    runtime_config: Optional[RuntimeConfig] = None,
    dp_config: Optional[DifferentialPrivacyConfig] = None
)
</code></pre>

Creates an TF optimizer from configurations.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`optimizer_config`
</td>
<td>
the parameters of the Optimization settings.
</td>
</tr><tr>
<td>
`runtime_config`
</td>
<td>
the parameters of the runtime.
</td>
</tr><tr>
<td>
`dp_config`
</td>
<td>
the parameter of differential privacy.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tf.optimizers.Optimizer object.
</td>
</tr>

</table>

<h3 id="inference_step"><code>inference_step</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>inference_step(
    inputs, model: tf.keras.Model
)
</code></pre>

Performs the forward step.

With distribution strategies, this method runs on devices.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
a dictionary of input tensors.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
the keras.Model.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Model outputs.
</td>
</tr>

</table>

<h3 id="initialize"><code>initialize</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initialize(
    model: tf.keras.Model
)
</code></pre>

[Optional] A callback function used as CheckpointManager's init_fn.

This function will be called when no checkpoint is found for the model. If there
is a checkpoint, the checkpoint will be loaded and this function will not be
called. You can use this callback function to load a pretrained checkpoint,
saved under a directory other than the model_dir.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
The keras.Model built or used by this task.
</td>
</tr>
</table>

<h3 id="process_compiled_metrics"><code>process_compiled_metrics</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>process_compiled_metrics(
    compiled_metrics, labels, model_outputs
)
</code></pre>

Process and update compiled_metrics.

call when using compile/fit API.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`compiled_metrics`
</td>
<td>
the compiled metrics (model.compiled_metrics).
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
a tensor or a nested structure of tensors.
</td>
</tr><tr>
<td>
`model_outputs`
</td>
<td>
a tensor or a nested structure of tensors. For example,
output of the keras model built by self.build_model.
</td>
</tr>
</table>

<h3 id="process_metrics"><code>process_metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L230-L232">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>process_metrics(
    metrics, labels, model_outputs
)
</code></pre>

Process and update metrics.

Called when using custom training loop API.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metrics`
</td>
<td>
a nested structure of metrics objects. The return of function
self.build_metrics.
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
a tensor or a nested structure of tensors.
</td>
</tr><tr>
<td>
`model_outputs`
</td>
<td>
a tensor or a nested structure of tensors. For example,
output of the keras model built by self.build_model.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
other args.
</td>
</tr>
</table>

<h3 id="reduce_aggregated_logs"><code>reduce_aggregated_logs</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reduce_aggregated_logs(
    aggregated_logs, global_step: Optional[tf.Tensor] = None
)
</code></pre>

Optional reduce of aggregated logs over validation steps.

This function reduces aggregated logs at the end of validation, and can be used
to compute the final metrics. It runs on CPU and in each eval_end() in base
trainer (see eval_end() function in official/core/base_trainer.py).

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`aggregated_logs`
</td>
<td>
Aggregated logs over multiple validation steps.
</td>
</tr><tr>
<td>
`global_step`
</td>
<td>
An optional variable of global step.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dictionary of reduced results.
</td>
</tr>

</table>

<h3 id="train_step"><code>train_step</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L234-L250">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train_step(
    inputs,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    metrics
)
</code></pre>

Does forward and backward.

With distribution strategies, this method runs on devices.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
a dictionary of input tensors.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
the model, forward pass definition.
</td>
</tr><tr>
<td>
`optimizer`
</td>
<td>
the optimizer for this training step.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
a nested structure of metrics objects.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dictionary of logs.
</td>
</tr>

</table>

<h3 id="validation_step"><code>validation_step</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L252-L263">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validation_step(
    inputs, model: tf.keras.Model, metrics=None
)
</code></pre>

Validation step.

With distribution strategies, this method runs on devices.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
a dictionary of input tensors.
</td>
</tr><tr>
<td>
`model`
</td>
<td>
the keras.Model.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
a nested structure of metrics objects.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dictionary of logs.
</td>
</tr>

</table>

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_name_scope(
    method
)
</code></pre>

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose names
included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method`
</td>
<td>
The method to wrap.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The original method wrapped such that it enters the module's name scope.
</td>
</tr>

</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
loss<a id="loss"></a>
</td>
<td>
`'loss'`
</td>
</tr>
</table>
