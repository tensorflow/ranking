description: Interface for feature preprocessing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.Preprocessor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
</div>

# tfr.keras.model.Preprocessor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L539-L585">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for feature preprocessing.

<!-- Placeholder for "Used in" -->

The `Preprocessor` class is an abstract class to implement `preprocess` in
`ModelBuilder` in tfr.keras.

To be implemented by subclasses:

*   `__call__()`: Contains the logic to preprocess context and example inputs.

Example subclass implementation:

```python
class SimplePreprocessor(Preprocessor):

  def __call__(self, context_inputs, example_inputs, mask):
    context_features = {
        name: tf.math.log1p(
            tf.abs(tensor)) for name, tensor in context_inputs.items()
    }
    example_features = {
        name: tf.math.log1p(
            tf.abs(tensor)) for name, tensor in example_inputs.items()
    }
    return context_features, example_features
```

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L567-L585">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__call__(
    context_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    example_inputs: <a href="../../../tfr/keras/model/TensorLike.md"><code>tfr.keras.model.TensorLike</code></a>,
    mask: tf.Tensor
) -> Tuple[TensorDict, TensorDict]
</code></pre>

Invokes the `Preprocessor` instance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context_inputs`
</td>
<td>
maps context feature keys to `tf.keras.Input`.
</td>
</tr><tr>
<td>
`example_inputs`
</td>
<td>
maps example feature keys to `tf.keras.Input`.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
[batch_size, list_size]-tensor of mask for valid examples.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of two dicts which map the context and example feature keys to
the corresponding `tf.Tensor`s.
</td>
</tr>

</table>
