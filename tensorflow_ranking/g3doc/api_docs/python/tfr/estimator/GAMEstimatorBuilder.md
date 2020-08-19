<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.estimator.GAMEstimatorBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="make_estimator"/>
</div>

# tfr.estimator.GAMEstimatorBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Builds a TFR estimator with subscore signatures of GAM models.

Inherits From: [`EstimatorBuilder`](../../tfr/estimator/EstimatorBuilder.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.estimator.GAMEstimatorBuilder(
    context_feature_columns, example_feature_columns, scoring_function,
    transform_function=None, optimizer=None, loss_reduction=None, hparams=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Neural Generalized Additive Ranking Model is an additive ranking model. See the
paper (https://arxiv.org/abs/2005.02553) for more details. For each example x
with n features (x_1, x_2, ..., x_n), the ranking score is:

F(x) = f1(x_1) + f2(x_2) + ... + fn(x_n)

where each feature is scored by a corresponding submodel, and the overall
ranking score is the sum of all the submodels' outputs. Each submodel is a
standalone feed-forward network.

When there are m context features (c_1, c_2, ..., c_m), the ranking score will
be determined by:

F(c, x) = w1(c) * f1(x_1) + w2(c) * f2(x_2) + ... + wn(c) * fn(x_n)

where (w1(c), w2(c), ..., wn(c)) is a weighting vector determined solely by
context features. For each context feature c_j, a feed-forward submodel is
constructed to derive a weighting vector (wj1(c_j), wj2(c_j), ..., wjn(c_j)).
The final weighting vector is the sum of the output of all the context features'
submodels.

The model is implicitly interpretable as the contribution of each feature to the
final ranking score can be easily visualized. However, the model does not have
higher-order inter-feature interactions and hence may not have performance as
good as the fully-connected DNN.

The output of each example feature's submodel can be retrieved by tensor named
`{feature_name}_subscore`. The output of each context feature's submodel is a
n-dimensional vector and can be retrieved by tensor named
`{feature_name}_subweight`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`context_feature_columns`
</td>
<td>
(dict) Context (aka, query) feature columns.
</td>
</tr><tr>
<td>
`example_feature_columns`
</td>
<td>
(dict) Example (aka, document) feature columns.
</td>
</tr><tr>
<td>
`scoring_function`
</td>
<td>
(function) A user-provided scoring function with the
below signatures:
* Args:
`context_features`: (dict) A dict of Tensors with the shape
[batch_size, ...].
`example_features`: (dict) A dict of Tensors with the shape
[batch_size, ...].
`mode`: (`estimator.ModeKeys`) Specifies if this is for training,
evaluation or inference. See ModeKeys.
* Returns: The computed logits, a Tensor of shape [batch_size, 1].
</td>
</tr><tr>
<td>
`transform_function`
</td>
<td>
(function) A user-provided function that transforms
raw features into dense Tensors with the following signature:
* Args:
`features`: (dict) A dict of Tensors or SparseTensors containing the
raw features from an `input_fn`.
`mode`: (`estimator.ModeKeys`) Specifies if this is for training,
evaluation or inference. See ModeKeys.
* Returns:
`context_features`: (dict) A dict of Tensors with the shape
[batch_size, ...].
`example_features`: (dict) A dict of Tensors with the shape
[batch_size, list_size, ...].
</td>
</tr><tr>
<td>
`optimizer`
</td>
<td>
(`tf.Optimizer`) An `Optimizer` object for model optimzation.
</td>
</tr><tr>
<td>
`loss_reduction`
</td>
<td>
(str) An enum of strings indicating the loss reduction
type. See type definition in the `tf.compat.v1.losses.Reduction`.
</td>
</tr><tr>
<td>
`hparams`
</td>
<td>
(dict) A dict containing model hyperparameters.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the `example_feature_columns` is None.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If the `scoring_function` is None..
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
If both the `optimizer` and the `hparams["learning_rate"]`
are not specified.
</td>
</tr>
</table>

## Methods

<h3 id="make_estimator"><code>make_estimator</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_estimator()
</code></pre>

Returns the built `tf.estimator.Estimator` for the TF-Ranking model.
