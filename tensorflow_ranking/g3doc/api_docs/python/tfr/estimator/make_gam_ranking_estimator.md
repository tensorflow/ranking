<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.estimator.make_gam_ranking_estimator" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.estimator.make_gam_ranking_estimator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds an `Estimator` instance with GAM scoring function.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.estimator.make_gam_ranking_estimator(
    example_feature_columns, example_hidden_units, context_feature_columns=None,
    context_hidden_units=None, optimizer=None, learning_rate=0.05,
    loss='approx_ndcg_loss',
    loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
    activation_fn=tf.nn.relu, dropout=None, use_batch_norm=False,
    batch_norm_moment=0.999, model_dir=None, checkpoint_secs=120,
    num_checkpoints=1000
)
</code></pre>

<!-- Placeholder for "Used in" -->

See the comment of `GAMEstimatorBuilder` class for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`example_feature_columns`
</td>
<td>
(dict) A dict containing all the example feature
columns used by the model. Keys are feature names, and values are
instances of classes derived from `_FeatureColumn`.
</td>
</tr><tr>
<td>
`example_hidden_units`
</td>
<td>
(list) Iterable of number hidden units per layer for
example features. All layers are fully connected. Ex. `[64, 32]` means
first layer has 64 nodes and second one has 32.
</td>
</tr><tr>
<td>
`context_feature_columns`
</td>
<td>
(dict) A dict containing all the context feature
columns used by the model. See `example_feature_columns`.
</td>
</tr><tr>
<td>
`context_hidden_units`
</td>
<td>
(list) Iterable of number hidden units per layer for
context features. See `example_hidden_units`.
</td>
</tr><tr>
<td>
`optimizer`
</td>
<td>
(`tf.Optimizer`) An `Optimizer` object for model optimzation. If
`None`, an Adagard optimizer with `learning_rate` will be created.
</td>
</tr><tr>
<td>
`learning_rate`
</td>
<td>
(float) Only used if `optimizer` is a string. Defaults to
0.05.
</td>
</tr><tr>
<td>
`loss`
</td>
<td>
(str) A string to decide the loss function used in training. See
`RankingLossKey` class for possible values.
</td>
</tr><tr>
<td>
`loss_reduction`
</td>
<td>
(str) An enum of strings indicating the loss reduction type.
See type definition in the `tf.compat.v1.losses.Reduction`.
</td>
</tr><tr>
<td>
`activation_fn`
</td>
<td>
Activation function applied to each layer. If `None`, will
use `tf.nn.relu`.
</td>
</tr><tr>
<td>
`dropout`
</td>
<td>
(float) When not `None`, the probability we will drop out a given
coordinate.
</td>
</tr><tr>
<td>
`use_batch_norm`
</td>
<td>
(bool) Whether to use batch normalization after each hidden
layer.
</td>
</tr><tr>
<td>
`batch_norm_moment`
</td>
<td>
(float) Momentum for the moving average in batch
normalization.
</td>
</tr><tr>
<td>
`model_dir`
</td>
<td>
(str) Directory to save model parameters, graph and etc. This can
also be used to load checkpoints from the directory into a estimator to
continue training a previously saved model.
</td>
</tr><tr>
<td>
`checkpoint_secs`
</td>
<td>
(int) Time interval (in seconds) to save checkpoints.
</td>
</tr><tr>
<td>
`num_checkpoints`
</td>
<td>
(int) Number of checkpoints to keep.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An `Estimator` with GAM scoring function.
</td>
</tr>

</table>
