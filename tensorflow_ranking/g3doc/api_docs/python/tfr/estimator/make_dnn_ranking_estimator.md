<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.estimator.make_dnn_ranking_estimator" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.estimator.make_dnn_ranking_estimator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Builds an `Estimator` instance with DNN scoring function.

```python
tfr.estimator.make_dnn_ranking_estimator(
    example_feature_columns, hidden_units, context_feature_columns=None,
    optimizer=None, learning_rate=0.05, listwise_inference=False,
    loss='approx_ndcg_loss',
    loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
    activation_fn=tf.nn.relu, dropout=None, use_batch_norm=False,
    batch_norm_moment=0.999, model_dir=None, checkpoint_secs=120,
    num_checkpoints=1000
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`example_feature_columns`</b>: (dict) Example (aka, document) feature
    columns.
*   <b>`hidden_units`</b>: (list) Iterable of number hidden units per layer for
    a DNN model. All layers are fully connected. Ex. `[64, 32]` means first
    layer has 64 nodes and second one has 32.
*   <b>`context_feature_columns`</b>: (dict) Context (aka, query) feature
    columns.
*   <b>`optimizer`</b>: (`tf.Optimizer`) An `Optimizer` object for model
    optimzation.
*   <b>`learning_rate`</b>: (float) Only used if `optimizer` is a string.
    Defaults to 0.05.
*   <b>`listwise_inference`</b>: (bool) Whether the inference will be performed
    with the listwise data format such as `ExampleListWithContext`.
*   <b>`loss`</b>: (str) A string to decide the loss function used in training.
    See `RankingLossKey` class for possible values.
*   <b>`loss_reduction`</b>: (str) An enum of strings indicating the loss
    reduction type. See type definition in the `tf.compat.v1.losses.Reduction`.
*   <b>`activation_fn`</b>: Activation function applied to each layer. If
    `None`, will use `tf.nn.relu`.
*   <b>`dropout`</b>: (float) When not `None`, the probability we will drop out
    a given coordinate.
*   <b>`use_batch_norm`</b>: (bool) Whether to use batch normalization after
    each hidden layer.
*   <b>`batch_norm_moment`</b>: (float) Momentum for the moving average in batch
    normalization.
*   <b>`model_dir`</b>: (str) Directory to save model parameters, graph and etc.
    This can also be used to load checkpoints from the directory into a
    estimator to continue training a previously saved model.
*   <b>`checkpoint_secs`</b>: (int) Time interval (in seconds) to save
    checkpoints.
*   <b>`num_checkpoints`</b>: (int) Number of checkpoints to keep.

#### Returns:

An `Estimator` with DNN scoring function.
