<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.losses.get" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.losses.get

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/losses.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Factory method to get a ranking loss class.

```python
tfr.keras.losses.get(
    loss, reduction=tf.losses.Reduction.AUTO, lambda_weight=None, name=None,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`loss`</b>: (str) An attribute of `RankingLossKey`, defining which loss
    object to return.
*   <b>`reduction`</b>: (enum) An enum of strings indicating the loss reduction
    type. See type definition in the `tf.compat.v2.losses.Reduction`.
*   <b>`lambda_weight`</b>: (losses_impl._LambdaWeight) A lambda object for
    ranking metric optimization.
*   <b>`name`</b>: (optional) (str) Name of loss.
*   <b>`**kwargs`</b>: Keyword arguments for the loss object.

#### Returns:

A ranking loss instance. See `_RankingLoss` signature for more details.

#### Raises:

*   <b>`ValueError`</b>: If loss_key is unsupported.
