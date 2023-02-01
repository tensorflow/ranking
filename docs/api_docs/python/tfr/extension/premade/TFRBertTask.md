description: Task object for tf-ranking BERT.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.premade.TFRBertTask" />
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

# tfr.extension.premade.TFRBertTask

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L138-L343">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Task object for tf-ranking BERT.

Inherits From: [`RankingTask`](../../../tfr/extension/task/RankingTask.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfr.extension.premade.tfrbert_task.TFRBertTask`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.extension.premade.TFRBertTask(
    params,
    label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
    logging_dir: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L220-L229">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>aggregate_logs(
    state=None, step_outputs=None
)
</code></pre>

Aggregates over logs. This runs on CPU in eager mode.

<h3 id="build_inputs"><code>build_inputs</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L188-L193">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_inputs(
    params, input_context=None
)
</code></pre>

Returns tf.data.Dataset for tf-ranking BERT task.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L163-L186">View
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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L328-L343">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initialize(
    model
)
</code></pre>

Load a pretrained checkpoint (if exists) and then train from iter 0.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L231-L259">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reduce_aggregated_logs(
    aggregated_logs, global_step=None
)
</code></pre>

Calculates aggregated metrics and writes predictions to csv.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/tfrbert_task.py#L195-L218">View
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
