description: Union of all types that can be converted to a tf.Tensor by
tf.convert_to_tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.TensorLike" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.model.TensorLike

<!-- Insert buttons and diff -->
This symbol is a **type alias**.

Union of all types that can be converted to a `tf.Tensor` by
`tf.convert_to_tensor`.

#### Source:

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TensorLike = Union[
    tensorflow.python.types.core.Tensor,
    tensorflow.python.types.core.TensorProtocol,
    int,
    float,
    bool,
    str,
    complex,
    tuple,
    list,
    numpy.ndarray
]
</code></pre>

<!-- Placeholder for "Used in" -->

This definition may be used in user code. Additional types may be added in the
future as more input types are supported.

#### Example:

```
def foo(x: TensorLike):
  pass
```

This definition passes static type verification for:

```
foo(tf.constant([1, 2, 3]))
foo([1, 2, 3])
foo(np.array([1, 2, 3]))
```
