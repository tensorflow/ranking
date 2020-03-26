<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.create_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.model.create_keras_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Creates a Functional Keras ranking model.

```python
tfr.keras.model.create_keras_model(
    network, loss, metrics, optimizer, size_feature_name
)
```

<!-- Placeholder for "Used in" -->

A mask is inferred from size_feature_name and passed to the network, along with
feature dictionary as inputs.

#### Args:

*   <b>`network`</b>:
    (<a href="../../../tfr/keras/network/RankingNetwork.md"><code>tfr.keras.network.RankingNetwork</code></a>)
    A ranking network which generates a list of scores.
*   <b>`loss`</b>: (`tfr.keras.losses._RankingLoss`) A ranking loss.
*   <b>`metrics`</b>: (list) List of ranking metrics,
    `tfr.keras.metrics._RankingMetric` instances.
*   <b>`optimizer`</b>: (`tf.keras.optimizer.Optimizer`) Optimizer to minimize
    ranking loss.
*   <b>`size_feature_name`</b>: (str) Name of feature for example list sizes. If
    not None, this feature name corresponds to a `tf.int32` Tensor of size
    [batch_size] corresponding to sizes of example lists. If `None`, all
    examples are treated as valid.

#### Returns:

A compiled ranking Keras model, a `tf.keras.Model` instance.
