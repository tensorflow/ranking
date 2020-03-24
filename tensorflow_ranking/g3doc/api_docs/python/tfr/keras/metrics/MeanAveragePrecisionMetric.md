<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.metrics.MeanAveragePrecisionMetric" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_metric"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="reset_states"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="update_state"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfr.keras.metrics.MeanAveragePrecisionMetric

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Implements mean average precision (MAP).

```python
tfr.keras.metrics.MeanAveragePrecisionMetric(
    name=None, topn=None, dtype=None, **kwargs
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`name`</b>: (Optional) string name of the metric instance.
*   <b>`dtype`</b>: (Optional) data type of the metric result.

#### Attributes:

*   <b>`activity_regularizer`</b>: Optional regularizer function for the output
    of this layer.
*   <b>`dtype`</b>: Dtype used by the weights of the layer, set in the
    constructor.
*   <b>`dynamic`</b>: Whether the layer is dynamic (eager-only); set in the
    constructor.
*   <b>`input`</b>: Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input, i.e. if it is connected
    to one incoming layer.

*   <b>`input_spec`</b>: `InputSpec` instance(s) describing the input format for
    this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called. Consider a
    `Conv2D` layer: it can only be called on a single input tensor of rank 4. As
    such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4 (for
    instance, an input of shape `(2,)`, it will raise a nicely-formatted error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:

    -   Structure (e.g. a single input, a list of 2 inputs, etc)
    -   Shape
    -   Rank (ndim)
    -   Dtype

    For more information, see `tf.keras.layers.InputSpec`.

*   <b>`losses`</b>: Losses which are associated with this `Layer`.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

*   <b>`metrics`</b>: List of `tf.keras.metrics.Metric` instances tracked by the
    layer.

*   <b>`name`</b>: Name of the layer (string), set in the constructor.

*   <b>`name_scope`</b>: Returns a `tf.name_scope` instance for this class.

*   <b>`non_trainable_weights`</b>: List of all non-trainable weights tracked by
    this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

*   <b>`output`</b>: Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output, i.e. if it is connected
    to one incoming layer.

*   <b>`submodules`</b>: Sequence of all sub-modules.

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

*   <b>`trainable`</b>

*   <b>`trainable_weights`</b>: List of all trainable weights tracked by this
    layer.

    Trainable weights are updated via gradient descent during training.

*   <b>`weights`</b>: Returns the list of all layer variables/weights.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(
    *args, **kwargs
)
```

Accumulates statistics and then computes metric result value.

#### Args:

*   <b>`*args`</b>: * <b>`**kwargs`</b>: A mini-batch of inputs to the Metric,
    passed on to `update_state()`.

#### Returns:

The metric value tensor.

<h3 id="add_loss"><code>add_loss</code></h3>

```python
add_loss(
    losses, inputs=None
)
```

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
    self.add_loss(tf.abs(tf.reduce_mean(inputs)), inputs=True)
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
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Weight regularization.
model.add_loss(lambda: tf.reduce_mean(x.kernel))
```

The `get_losses_for` method allows to retrieve the losses relevant to a specific
set of inputs.

#### Arguments:

*   <b>`losses`</b>: Loss tensor, or list/tuple of tensors. Rather than tensors,
    losses may also be zero-argument callables which create a loss tensor.
*   <b>`inputs`</b>: Ignored when executing eagerly. If anything other than None
    is passed, it signals the losses are conditional on some of the layer's
    inputs, and thus they should only be run where these inputs are available.
    This is the case for activity regularization losses, for instance. If `None`
    is passed, the losses are assumed to be unconditional, and will apply across
    all dataflows of the layer (e.g. weight regularization losses).

<h3 id="add_metric"><code>add_metric</code></h3>

```python
add_metric(
    value, aggregation=None, name=None
)
```

Adds metric tensor to the layer.

#### Args:

*   <b>`value`</b>: Metric tensor.
*   <b>`aggregation`</b>: Sample-wise metric reduction function. If
    `aggregation=None`, it indicates that the metric tensor provided has been
    aggregated already. eg, `bin_acc = BinaryAccuracy(name='acc')` followed by
    `model.add_metric(bin_acc(y_true, y_pred))`. If aggregation='mean', the
    given metric tensor will be sample-wise reduced using `mean` function. eg,
    `model.add_metric(tf.reduce_sum(outputs), name='output_mean',
    aggregation='mean')`.
*   <b>`name`</b>: String metric name.

#### Raises:

*   <b>`ValueError`</b>: If `aggregation` is anything other than None or `mean`.

<h3 id="build"><code>build</code></h3>

```python
build(
    input_shape
)
```

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model` can
override if they need a state-creation step in-between layer instantiation and
layer call.

This is typically used to create the weights of `Layer` subclasses.

#### Arguments:

*   <b>`input_shape`</b>: Instance of `TensorShape`, or list of instances of
    `TensorShape` if the layer expects a list of inputs (one instance per
    input).

<h3 id="compute_mask"><code>compute_mask</code></h3>

```python
compute_mask(
    inputs, mask=None
)
```

Computes an output mask tensor.

#### Arguments:

*   <b>`inputs`</b>: Tensor or list of tensors.
*   <b>`mask`</b>: Tensor or list of tensors.

#### Returns:

None or a tensor (or list of tensors, one per output tensor of the layer).

<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

```python
compute_output_shape(
    input_shape
)
```

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the layer.
This assumes that the layer will later be used with inputs that match the input
shape provided here.

#### Arguments:

*   <b>`input_shape`</b>: Shape tuple (tuple of integers) or list of shape
    tuples (one per output tensor of the layer). Shape tuples can include None
    for free dimensions, instead of an integer.

#### Returns:

An input shape tuple.

<h3 id="count_params"><code>count_params</code></h3>

```python
count_params()
```

Count the total number of scalars composing the weights.

#### Returns:

An integer count.

#### Raises:

*   <b>`ValueError`</b>: if the layer isn't yet built (in which case its weights
    aren't yet defined).

<h3 id="from_config"><code>from_config</code></h3>

```python
@classmethod
from_config(
    cls, config
)
```

Creates a layer from its config.

This method is the reverse of `get_config`, capable of instantiating the same
layer from the config dictionary. It does not handle layer connectivity (handled
by Network), nor weights (handled by `set_weights`).

#### Arguments:

*   <b>`config`</b>: A Python dictionary, typically the output of get_config.

#### Returns:

A layer instance.

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/metrics.py">View
source</a>

```python
get_config()
```

Returns the serializable config of the metric.

<h3 id="get_weights"><code>get_weights</code></h3>

```python
get_weights()
```

Returns the current weights of the layer.

The weights of a layer represent the state of the layer. This function returns
both trainable and non-trainable weight values associated with this layer as a
list of Numpy arrays, which can in turn be used to load state into similarly
parameterized layers.

For example, a Dense layer returns a list of two values-- per-output weights and
the bias value. These can be used to set the weights of another Dense layer:

```
>>> a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> b.set_weights(a.get_weights())
>>> b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

#### Returns:

Weights values as a list of numpy arrays.

<h3 id="reset_states"><code>reset_states</code></h3>

```python
reset_states()
```

Resets all of the metric state variables.

This function is called between epochs/steps, when a metric is evaluated during
training.

<h3 id="result"><code>result</code></h3>

```python
result()
```

Computes and returns the metric value tensor.

Result computation is an idempotent operation that simply calculates the metric
value using the state variables.

<h3 id="set_weights"><code>set_weights</code></h3>

```python
set_weights(
    weights
)
```

Sets the weights of the layer, from Numpy arrays.

The weights of a layer represent the state of the layer. This function sets the
weight values from numpy arrays. The weight values should be passed in the order
they are created by the layer. Note that the layer's weights must be
instantiated before calling this function by calling the layer.

For example, a Dense layer returns a list of two values-- per-output weights and
the bias value. These can be used to set the weights of another Dense layer:

```
>>> a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> b.set_weights(a.get_weights())
>>> b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

#### Arguments:

*   <b>`weights`</b>: a list of Numpy arrays. The number of arrays and their
    shape must match number of the dimensions of the weights of the layer (i.e.
    it should match the output of `get_weights`).

#### Raises:

*   <b>`ValueError`</b>: If the provided weights list does not match the layer's
    specifications.

<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/metrics.py">View
source</a>

```python
update_state(
    y_true, y_pred, sample_weight=None
)
```

Accumulates metric statistics.

`y_true` and `y_pred` should have the same shape.

#### Args:

*   <b>`y_true`</b>: The ground truth values.
*   <b>`y_pred`</b>: The predicted values.
*   <b>`sample_weight`</b>: Optional weighting of each example. Defaults to 1.
    Can be a `Tensor` whose rank is either 0, or the same rank as `y_true`, and
    must be broadcastable to `y_true`.

#### Returns:

Update op.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

```python
@classmethod
with_name_scope(
    cls, method
)
```

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

#### Args:

*   <b>`method`</b>: The method to wrap.

#### Returns:

The original method wrapped such that it enters the module's name scope.
