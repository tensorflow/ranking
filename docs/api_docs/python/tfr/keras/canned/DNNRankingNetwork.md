description: Deep Neural Network (DNN) scoring based univariate ranking network.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.canned.DNNRankingNetwork" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_metric"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_logits"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="score"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="transform"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfr.keras.canned.DNNRankingNetwork

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/canned/dnn.py#L9-L123">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Deep Neural Network (DNN) scoring based univariate ranking network.

Inherits From:
[`UnivariateRankingNetwork`](../../../tfr/keras/network/UnivariateRankingNetwork.md),
[`RankingNetwork`](../../../tfr/keras/network/RankingNetwork.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfr.keras.canned.dnn.DNNRankingNetwork`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.canned.DNNRankingNetwork(
    context_feature_columns=None, example_feature_columns=None,
    hidden_layer_dims=None, activation=None, use_batch_norm=True,
    batch_norm_moment=0.999, dropout=0.5, name=&#x27;dnn_ranking_network&#x27;,
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
`context_feature_columns`
</td>
<td>
A dict containing all the context feature columns
used by the network. Keys are feature names, and values are instances of
classes derived from `_FeatureColumn`.
</td>
</tr><tr>
<td>
`example_feature_columns`
</td>
<td>
A dict containing all the example feature columns
used by the network. Keys are feature names, and values are instances of
classes derived from `_FeatureColumn`.
</td>
</tr><tr>
<td>
`hidden_layer_dims`
</td>
<td>
Iterable of number hidden units per layer. All layers
are fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
second one has 32.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Activation function applied to each layer. If `None`, will use
an identity activation, which is default behavior in Keras activations.
</td>
</tr><tr>
<td>
`use_batch_norm`
</td>
<td>
Whether to use batch normalization after each hidden
layer.
</td>
</tr><tr>
<td>
`batch_norm_moment`
</td>
<td>
Momentum for the moving average in batch normalization.
</td>
</tr><tr>
<td>
`dropout`
</td>
<td>
When not `None`, the probability we will drop out a given
coordinate.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
name of Keras network.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
`ValueError` if `example_feature_columns` or `hidden_layer_dims` is empty.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `activity_regularizer` </td> <td> Optional regularizer function for
the output of this layer. </td> </tr><tr> <td> `compute_dtype` </td> <td> The
dtype of the layer's computations.

This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless mixed precision
is used, this is the same as `Layer.dtype`, the dtype of the weights.

Layers automatically cast their inputs to the compute dtype, which causes
computations and the output to be in the compute dtype as well. This is done by
the base Layer class in `Layer.__call__`, so you do not have to insert these
casts if implementing your own layer.

Layers often perform certain internal computations in higher precision when
`compute_dtype` is float16 or bfloat16 for numeric stability. The output will
still typically be float16 or bfloat16 in such cases. </td> </tr><tr> <td>
`context_feature_columns` </td> <td>

</td> </tr><tr> <td> `dtype` </td> <td> The dtype of the layer weights.

This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless mixed
precision is used, this is the same as `Layer.compute_dtype`, the dtype of the
layer's computations. </td> </tr><tr> <td> `dtype_policy` </td> <td> The dtype
policy associated with this layer.

This is an instance of a `tf.keras.mixed_precision.Policy`. </td> </tr><tr> <td>
`dynamic` </td> <td> Whether the layer is dynamic (eager-only); set in the
constructor. </td> </tr><tr> <td> `example_feature_columns` </td> <td>

</td> </tr><tr> <td> `input` </td> <td> Retrieves the input tensor(s) of a
layer.

Only applicable if the layer has exactly one input, i.e. if it is connected to
one incoming layer. </td> </tr><tr> <td> `input_spec` </td> <td> `InputSpec`
instance(s) describing the input format for this layer.

When you create a layer subclass, you can set `self.input_spec` to enable the
layer to run input compatibility checks when it is called. Consider a `Conv2D`
layer: it can only be called on a single input tensor of rank 4. As such, you
can set, in `__init__()`:

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
```

Now, if you try to call the layer on an input that isn't rank 4 (for instance,
an input of shape `(2,)`, it will raise a nicely-formatted error:

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
```

Input checks that can be specified via `input_spec` include: - Structure (e.g. a
single input, a list of 2 inputs, etc) - Shape - Rank (ndim) - Dtype

For more information, see `tf.keras.layers.InputSpec`. </td> </tr><tr> <td>
`losses` </td> <td> List of losses added using the `add_loss()` API.

Variable regularization tensors are created when this property is accessed, so
it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate
gradients back to the corresponding variables.

```
>>> class MyLayer(tf.keras.layers.Layer):
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
>>> l = MyLayer()
>>> l(np.ones((10, 1)))
>>> l.losses
[1.0]
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> x = tf.keras.layers.Dense(10)(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Activity regularization.
>>> len(model.losses)
0
>>> model.add_loss(tf.abs(tf.reduce_mean(x)))
>>> len(model.losses)
1
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
>>> x = d(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Weight regularization.
>>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
>>> model.losses
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
```

</td> </tr><tr> <td> `metrics` </td> <td> List of metrics added using the
`add_metric()` API.

```
>>> input = tf.keras.layers.Input(shape=(3,))
>>> d = tf.keras.layers.Dense(2)
>>> output = d(input)
>>> d.add_metric(tf.reduce_max(output), name='max')
>>> d.add_metric(tf.reduce_min(output), name='min')
>>> [m.name for m in d.metrics]
['max', 'min']
```

</td> </tr><tr> <td> `name` </td> <td> Name of the layer (string), set in the
constructor. </td> </tr><tr> <td> `name_scope` </td> <td> Returns a
`tf.name_scope` instance for this class. </td> </tr><tr> <td>
`non_trainable_weights` </td> <td> List of all non-trainable weights tracked by
this layer.

Non-trainable weights are *not* updated during training. They are expected to be
updated manually in `call()`. </td> </tr><tr> <td> `output` </td> <td> Retrieves
the output tensor(s) of a layer.

Only applicable if the layer has exactly one output, i.e. if it is connected to
one incoming layer. </td> </tr><tr> <td> `submodules` </td> <td> Sequence of all
sub-modules.

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

</td> </tr><tr> <td> `supports_masking` </td> <td> Whether this layer supports
computing a mask using `compute_mask`. </td> </tr><tr> <td> `trainable` </td>
<td>

</td> </tr><tr> <td> `trainable_weights` </td> <td> List of all trainable
weights tracked by this layer.

Trainable weights are updated via gradient descent during training.
</td>
</tr><tr>
<td>
`variable_dtype`
</td>
<td>
Alias of `Layer.dtype`, the dtype of the weights.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Returns the list of all layer variables/weights.
</td>
</tr>
</table>

## Methods

<h3 id="add_loss"><code>add_loss</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_loss(
    losses, **kwargs
)
</code></pre>

Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be dependent on
the inputs passed when calling a layer. Hence, when reusing the same layer on
different inputs `a` and `b`, some entries in `layer.losses` may be dependent on
`a` and some on `b`. This method automatically keeps track of dependencies.

This method can be used inside a subclassed layer or model's `call` function, in
which case `losses` should be a Tensor or list of Tensors.

#### Example:

```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any loss Tensors passed to this Model must be
symbolic and be able to be traced back to the model's `Input`s. These losses
become part of the model's topology and are tracked in `get_config`.

#### Example:

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

If this is not the case for your loss (if, for example, your loss references a
`Variable` of one of the model's layers), you can wrap your loss in a
zero-argument lambda. These losses are not tracked as part of the model's
topology since they can't be serialized.

#### Example:

```python
inputs = tf.keras.Input(shape=(10,))
d = tf.keras.layers.Dense(10)
x = d(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Weight regularization.
model.add_loss(lambda: tf.reduce_mean(d.kernel))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`losses`
</td>
<td>
Loss tensor, or list/tuple of tensors. Rather than tensors, losses
may also be zero-argument callables which create a loss tensor.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
inputs - Deprecated, will be automatically inferred.
</td>
</tr>
</table>

<h3 id="add_metric"><code>add_metric</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_metric(
    value, name=None, **kwargs
)
</code></pre>

Adds metric tensor to the layer.

This method can be used inside the `call()` method of a subclassed layer or
model.

```python
class MyMetricLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

  def call(self, inputs):
    self.add_metric(self.mean(inputs))
    self.add_metric(tf.reduce_sum(inputs), name='metric_2')
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any tensor passed to this Model must be symbolic and
be able to be traced back to the model's `Input`s. These metrics become part of
the model's topology and are tracked when you save the model via `save()`.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(math_ops.reduce_sum(x), name='metric_1')
```

Note: Calling `add_metric()` with the result of a metric object on a Functional
Model, as shown in the example below, is not supported. This is because we
cannot trace the metric result tensor back to the model's inputs.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
Metric tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String metric name.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
`aggregation` - When the `value` tensor provided is not the result of
calling a `keras.Metric` instance, it will be aggregated by default
using a `keras.Metric.Mean`.
</td>
</tr>
</table>

<h3 id="build"><code>build</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    input_shape
)
</code></pre>

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model` can
override if they need a state-creation step in-between layer instantiation and
layer call.

This is typically used to create the weights of `Layer` subclasses.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Instance of `TensorShape`, or list of instances of
`TensorShape` if the layer expects a list of inputs
(one instance per input).
</td>
</tr>
</table>

<h3 id="compute_logits"><code>compute_logits</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/network.py#L252-L289">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_logits(
    context_features=None, example_features=None, training=None, mask=None
)
</code></pre>

Scores context and examples to return a score per document.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_features`
</td>
<td>
(dict) context feature names to 2D tensors of shape
[batch_size, feature_dims].
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
(dict) example feature names to 3D tensors of shape
[batch_size, list_size, feature_dims].
</td>
</tr><tr>
<td>
`training`
</td>
<td>
(bool) whether in train or inference mode.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
(tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
is True for a valid example and False for invalid one. If mask is None,
all entries are valid.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
(tf.Tensor) A score tensor of shape [batch_size, list_size].
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If `scorer` does not return a scalar output.
</td>
</tr>
</table>

<h3 id="compute_mask"><code>compute_mask</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_mask(
    inputs, mask=None
)
</code></pre>

Computes an output mask tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Tensor or list of tensors.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
Tensor or list of tensors.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a tensor (or list of tensors,
one per output tensor of the layer).
</td>
</tr>

</table>

<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_output_shape(
    input_shape
)
</code></pre>

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the layer.
This assumes that the layer will later be used with inputs that match the input
shape provided here.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Shape tuple (tuple of integers)
or list of shape tuples (one per output tensor of the layer).
Shape tuples can include None for free dimensions,
instead of an integer.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An input shape tuple.
</td>
</tr>

</table>

<h3 id="count_params"><code>count_params</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>count_params()
</code></pre>

Count the total number of scalars composing the weights.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer count.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer isn't yet built
(in which case its weights aren't yet defined).
</td>
</tr>
</table>

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/network.py#L198-L216">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config, custom_objects=None
)
</code></pre>

Creates a RankingNetwork layer from its config.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
(dict) Layer configuration, typically the output of `get_config`.
</td>
</tr><tr>
<td>
`custom_objects`
</td>
<td>
(dict) Optional dictionary mapping names to custom classes
or functions to be considered during deserialization.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A RankingNetwork layer.
</td>
</tr>

</table>

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/canned/dnn.py#L114-L123">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config of the layer.

A layer config is a Python dictionary (serializable) containing the
configuration of a layer. The same layer can be reinstantiated later (without
its trained weights) from this configuration.

The config of a layer does not include connectivity information, nor the layer
class name. These are handled by `Network` (one layer of abstraction above).

Note that `get_config()` does not guarantee to return a fresh copy of dict every
time it is called. The callers should make a copy of the returned dict if they
want to modify it.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary.
</td>
</tr>

</table>

<h3 id="get_weights"><code>get_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weights()
</code></pre>

Returns the current weights of the layer, as NumPy arrays.

The weights of a layer represent the state of the layer. This function returns
both trainable and non-trainable weight values associated with this layer as a
list of NumPy arrays, which can in turn be used to load state into similarly
parameterized layers.

For example, a `Dense` layer returns a list of two values: the kernel matrix and
the bias vector. These can be used to set the weights of another `Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Weights values as a list of NumPy arrays.
</td>
</tr>

</table>

<h3 id="score"><code>score</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/canned/dnn.py#L84-L112">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>score(
    context_features=None, example_features=None, training=True
)
</code></pre>

Univariate scoring of context and one example to generate a score.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_features`
</td>
<td>
(dict) context feature names to 2D tensors of shape
[batch_size, ...].
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
(dict) example feature names to 2D tensors of shape
[batch_size, ...].
</td>
</tr><tr>
<td>
`training`
</td>
<td>
(bool) whether in training or inference mode.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
(tf.Tensor) A score tensor of shape [batch_size, 1].
</td>
</tr>

</table>

<h3 id="set_weights"><code>set_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>

Sets the weights of the layer, from NumPy arrays.

The weights of a layer represent the state of the layer. This function sets the
weight values from numpy arrays. The weight values should be passed in the order
they are created by the layer. Note that the layer's weights must be
instantiated before calling this function, by calling the layer.

For example, a `Dense` layer returns a list of two values: the kernel matrix and
the bias vector. These can be used to set the weights of another `Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`weights`
</td>
<td>
a list of NumPy arrays. The number
of arrays and their shape must match
number of the dimensions of the weights
of the layer (i.e. it should match the
output of `get_weights`).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided weights list does not match the
layer's specifications.
</td>
</tr>
</table>

<h3 id="transform"><code>transform</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/network.py#L118-L141">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transform(
    features=None, training=None, mask=None
)
</code></pre>

Transforms the features into dense context features and example features.

The user can overwrite this function for custom transformations. Mask is
provided as an argument so that inherited models can have access to it for
custom feature transformations, without modifying `call` explicitly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`features`
</td>
<td>
(dict) with a mix of context (2D) and example features (3D).
</td>
</tr><tr>
<td>
`training`
</td>
<td>
(bool) whether in train or inference mode.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
(tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
is True for a valid example and False for invalid one.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`context_features`
</td>
<td>
(dict) context feature names to dense 2D tensors of
shape [batch_size, feature_dims].
</td>
</tr><tr>
<td>
`example_features`
</td>
<td>
(dict) example feature names to dense 3D tensors of
shape [batch_size, list_size, feature_dims].
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

<h3 id="__call__"><code>__call__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Wraps `call`, applying pre- and post-processing steps.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
Positional arguments to be passed to `self.call`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to be passed to `self.call`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Output tensor(s).
</td>
</tr>

</table>

#### Note:

-   The following optional keyword arguments are reserved for specific uses:
    *   `training`: Boolean scalar tensor of Python boolean indicating whether
        the `call` is meant for training or inference.
    *   `mask`: Boolean input mask.
-   If the layer's `call` method takes a `mask` argument (as some Keras layers
    do), its default value will be set to the mask generated for `inputs` by the
    previous layer (if `input` did come from a layer that generated a
    corresponding mask, i.e. if it came from a Keras layer with masking support.
-   If the layer is not built, the method will call `build`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer's `call` method returns None (an invalid value).
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
if `super().__init__()` was not called in the constructor.
</td>
</tr>
</table>
