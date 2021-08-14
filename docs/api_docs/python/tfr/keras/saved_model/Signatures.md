description: Defines signatures to support regress and predict serving.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.saved_model.Signatures" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="normalize_outputs"/>
<meta itemprop="property" content="predict_tf_function"/>
<meta itemprop="property" content="regress_tf_function"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfr.keras.saved_model.Signatures

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/saved_model.py#L11-L162">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Defines signatures to support regress and predict serving.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.saved_model.Signatures(
    model: tf.keras.Model,
    context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.RaggedFeature]],
    example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature, tf.io.RaggedFeature]],
    mask_feature_name: str
)
</code></pre>

<!-- Placeholder for "Used in" -->

This wraps the trained Keras model in two serving functions that can be saved
with `tf.saved_model.save` or `model.save`, and loaded with corresponding
signature names. The regress serving signature takes a batch of serialized
`tf.Example`s as input, whereas the predict serving signature takes a batch of
serialized `ExampleListWithContext` as input.

#### Example usage:

A ranking model can be saved with signatures as follows:

```python
tf.saved_model.save(model, path, signatures=Signatures(model, ...)())
```

For regress serving, scores can be generated using `REGRESS` signature as
follows:

```python
loaded_model = tf.saved_model.load(path)
predictor = loaded_model.signatures[tf.saved_model.REGRESS_METHOD_NAME]
scores = predictor(serialized_examples)[tf.saved_model.REGRESS_OUTPUTS]
```

For predict serving, scores can be generated using `PREDICT` signature as
follows:

```python
loaded_model = tf.saved_model.load(path)
predictor = loaded_model.signatures[tf.saved_model.PREDICT_METHOD_NAME]
scores = predictor(serialized_elwcs)[tf.saved_model.PREDICT_OUTPUTS]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`
</td>
<td>
A Keras ranking model.
</td>
</tr><tr>
<td>
`context_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature` or `RaggedFeature` values for context in
`tensorflow.serving.ExampleListWithContext` proto.
</td>
</tr><tr>
<td>
`example_feature_spec`
</td>
<td>
(dict) A mapping from feature keys to
`FixedLenFeature` or `Ragged` values for examples in
`tensorflow.serving.ExampleListWithContext` proto.
</td>
</tr><tr>
<td>
`mask_feature_name`
</td>
<td>
(str) Name of feature for example list masks.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `name` </td> <td> Returns the name of this module as passed or
determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes parent
module names. </td> </tr><tr> <td> `name_scope` </td> <td> Returns a
`tf.name_scope` instance for this class. </td> </tr><tr> <td>
`non_trainable_variables` </td> <td> Sequence of non-trainable variables owned
by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`submodules` </td> <td> Sequence of all sub-modules.

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

</td> </tr><tr> <td> `trainable_variables` </td> <td> Sequence of trainable
variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`variables` </td> <td> Sequence of variables owned by this module and its
submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>

## Methods

<h3 id="normalize_outputs"><code>normalize_outputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/saved_model.py#L71-L92">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>normalize_outputs(
    default_key: str,
    outputs: Union[tf.Tensor, Dict[str, tf.Tensor]]
) -> Dict[str, tf.Tensor]
</code></pre>

Returns a dict of Tensors for outputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`default_key`
</td>
<td>
If outputs is a Tensor, use the default_key to make a dict.
</td>
</tr><tr>
<td>
`outputs`
</td>
<td>
outputs to be normalized.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict maps from str-like key(s) to Tensor(s).
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
TypeError if outputs is not a Tensor nor a dict.
</td>
</tr>

</table>

<h3 id="predict_tf_function"><code>predict_tf_function</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/saved_model.py#L94-L111">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict_tf_function() -> Callable[[tf.Tensor], Dict[str, tf.Tensor]]
</code></pre>

Makes a tensorflow function for `predict`.

<h3 id="regress_tf_function"><code>regress_tf_function</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/saved_model.py#L113-L132">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>regress_tf_function() -> Callable[[tf.Tensor], Dict[str, tf.Tensor]]
</code></pre>

Makes a tensorflow function for `regress`.

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

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/saved_model.py#L134-L162">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    serving_default: str = &#x27;regress&#x27;
) -> Dict[str, Callable[[tf.Tensor], Dict[str, tf.Tensor]]]
</code></pre>

Returns a dict of signatures.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`serving_default`
</td>
<td>
Specifies "regress" or "predict" as the serving_default
signature.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict of signatures.
</td>
</tr>

</table>
