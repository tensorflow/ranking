description: Interface for input creator.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.InputCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
</div>

# tfr.keras.model.InputCreator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L385-L414">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for input creator.

<!-- Placeholder for "Used in" -->

The `InputCreator` class is an abstract class to implement `create_inputs` in
`ModelBuilder` in tfr.keras.

To be implemented by subclasses:

*   `__call__()`: Contains the logic to create `tf.keras.Input` for context and
    example inputs.

Example subclass implementation:

```python
class SimpleInputCreator(InputCreator):

  def __call__(self):
    return {}, {"example_feature_1": tf.keras.Input((1,), dtype=tf.float32)}
```

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L406-L414">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__call__() -> Tuple[TensorDict, TensorDict]
</code></pre>

Invokes the `InputCreator` instance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of two dicts which map the context and example feature keys to
the corresponding `tf.keras.Input`.
</td>
</tr>

</table>
