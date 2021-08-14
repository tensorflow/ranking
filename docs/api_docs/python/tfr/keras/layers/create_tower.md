description: Creates a feed-forward network as tf.keras.Sequential.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.layers.create_tower" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.layers.create_tower

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/layers.py#L13-L64">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates a feed-forward network as `tf.keras.Sequential`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.layers.create_tower(
    hidden_layer_dims: List[int],
    output_units: int,
    activation: Optional[Callable[..., tf.Tensor]] = None,
    input_batch_norm: bool = False,
    use_batch_norm: bool = True,
    batch_norm_moment: float = 0.999,
    dropout: float = 0.5,
    name: Optional[str] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

It creates a feed-forward network with batch normalization and dropout, and
optionally applies batch normalization on inputs.

#### Example usage:

```python
tower = create_tower(hidden_layer_dims=[64, 32, 16], output_units=1)
inputs = tf.ones([2, 3, 1])
tower_logits = tower(inputs)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`hidden_layer_dims`
</td>
<td>
Iterable of number hidden units per layer. All layers are
fully connected. Ex. `[64, 32]` means first layer has 64 nodes and second
one has 32.
</td>
</tr><tr>
<td>
`output_units`
</td>
<td>
Size of output logits from this tower.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Activation function applied to each layer. If `None`, will use
an identity activation.
</td>
</tr><tr>
<td>
`input_batch_norm`
</td>
<td>
Whether to use batch normalization for input layer.
</td>
</tr><tr>
<td>
`use_batch_norm`
</td>
<td>
Whether to use batch normalization after each hidden layer.
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
Name of the Keras layer.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments for every `tf.keras.Dense` layers.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `tf.keras.Sequential` object.
</td>
</tr>

</table>
