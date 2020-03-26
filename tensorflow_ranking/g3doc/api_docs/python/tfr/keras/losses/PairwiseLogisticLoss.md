<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.PairwiseLogisticLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfr.keras.losses.PairwiseLogisticLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

For pairwise logistic loss.

```python
tfr.keras.losses.PairwiseLogisticLoss(
    reduction=tf.losses.Reduction.AUTO, name=None, lambda_weight=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`reduction`</b>: (Optional) Type of `tf.keras.losses.Reduction` to apply
    to loss. Default value is `AUTO`. `AUTO` indicates that the reduction option
    will be determined by the usage context. For almost all cases this defaults
    to `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside
    of built-in training loops such as `tf.keras` `compile` and `fit`, using
    `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
    training
    [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)
    for more details.
*   <b>`name`</b>: Optional name for the op.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py">View
source</a>

```python
__call__(
    y_true, y_pred, sample_weight=None
)
```

See tf.keras.losses.Loss.

<h3 id="from_config"><code>from_config</code></h3>

```python
@classmethod
from_config(
    cls, config
)
```

Instantiates a `Loss` from its config (output of `get_config()`).

#### Args:

*   <b>`config`</b>: Output of `get_config()`.

#### Returns:

A `Loss` instance.

<h3 id="get_config"><code>get_config</code></h3>

```python
get_config()
```

Returns the config dictionary for a `Loss` instance.
