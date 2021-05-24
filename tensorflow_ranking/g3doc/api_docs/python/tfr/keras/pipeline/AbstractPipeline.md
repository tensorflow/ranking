description: Interface for ranking pipeline to train a tf.keras.Model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline.AbstractPipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="build_loss"/>
<meta itemprop="property" content="build_metrics"/>
<meta itemprop="property" content="build_weighted_metrics"/>
<meta itemprop="property" content="train_and_validate"/>
</div>

# tfr.keras.pipeline.AbstractPipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L19-L143">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Interface for ranking pipeline to train a `tf.keras.Model`.

<!-- Placeholder for "Used in" -->

The `AbstractPipeline` class is an abstract class to train and validate a
ranking model in tfr.keras.

To be implemented by subclasses:

*   `build_loss()`: Contains the logic to build a `tf.keras.losses.Loss` or a
    dict or list of `tf.keras.losses.Loss`s to be optimized in training.
*   `build_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s to monitor and evaluate the training.
*   `build_weighted_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s which will take the weights.
*   `train_and_validate()`: Contrains the main training pipeline for training
    and validation.

Example subclass implementation:

```python
class BasicPipeline(AbstractPipeline):

  def __init__(self, model, train_data, valid_data, name=None):
    self._model = model
    self._train_data = train_data
    self._valid_data = valid_data
    self._name = name

  def build_loss(self):
    return tfr.keras.losses.get('softmax_loss')

  def build_metrics(self):
    return [
        tfr.keras.metrics.get(
            'ndcg', topn=topn, name='ndcg_{}'.format(topn)
        ) for topn in [1, 5, 10]
    ]

  def build_weighted_metrics(self):
    return [
        tfr.keras.metrics.get(
            'ndcg', topn=topn, name='weighted_ndcg_{}'.format(topn)
        ) for topn in [1, 5, 10]
    ]

  def train_and_validate(self, *arg, **kwargs):
    self._model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=self.build_loss(),
        metrics=self.build_metrics(),
        weighted_metrics=self.build_weighted_metrics())
    self._model.fit(
        x=self._train_data,
        epochs=100,
        validation_data=self._valid_data)
```

## Methods

<h3 id="build_loss"><code>build_loss</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L77-L91">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_loss() -> Any
</code></pre>

Returns the loss for model.compile.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
loss = pipeline.build_loss()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.keras.losses.Loss` or a dict or list of `tf.keras.losses.Loss`.
</td>
</tr>

</table>

<h3 id="build_metrics"><code>build_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L93-L107">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_metrics() -> Any
</code></pre>

Returns a list of ranking metrics for `model.compile()`.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
metrics = pipeline.build_metrics()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list or a dict of `tf.keras.metrics.Metric`s.
</td>
</tr>

</table>

<h3 id="build_weighted_metrics"><code>build_weighted_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L109-L123">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build_weighted_metrics() -> Any
</code></pre>

Returns a list of weighted ranking metrics for model.compile.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
weighted_metrics = pipeline.build_weighted_metrics()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list or a dict of `tf.keras.metrics.Metric`s.
</td>
</tr>

</table>

<h3 id="train_and_validate"><code>train_and_validate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py#L125-L143">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>train_and_validate(
    *arg, **kwargs
) -> Any
</code></pre>

Constructs and runs the training pipeline.

#### Example usage:

```python
pipeline = BasicPipeline(model, train_data, valid_data)
pipeline.train_and_validate()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*arg`
</td>
<td>
arguments that might be used in the training pipeline.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments that might be used in the training pipeline.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a trained `tf.keras.Model` or a path to a saved `tf.keras.Model`.
</td>
</tr>

</table>
