description: Interface to build a tf.keras.Model for ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.AbstractModelBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="build"/>
</div>

# tfr.keras.model.AbstractModelBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L67-L97">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface to build a `tf.keras.Model` for ranking.

<!-- Placeholder for "Used in" -->

The `AbstractModelBuilder` serves as the interface between model building and
training. The training pipeline just calls the `build()` method to get the model
constructed in the strategy scope used in the training pipeline, so for all
variables in the model, optimizers, and metrics. See `ModelFitPipeline` in
`pipeline.py` for example.

The `build()` method is to be implemented in a subclass. The simplest example is
just to define everything inside the build function when you define a
tf.keras.Model.

```python
class MyModelBuilder(AbstractModelBuilder):

  def build(self) -> tf.keras.Model:
    inputs = ...
    outputs = ...
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

The `MyModelBuilder` should work with `ModelFitPipeline`. To make the model
building more structured for ranking problems, we also define subclasses like
`ModelBuilderWithMask` in the following.

## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L94-L97">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build() -> tf.keras.Model
</code></pre>

The build method to be implemented by a subclass.
