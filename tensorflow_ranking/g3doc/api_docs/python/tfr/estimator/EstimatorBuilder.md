<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.estimator.EstimatorBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="make_estimator"/>
</div>

# tfr.estimator.EstimatorBuilder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Builds a tf.estimator.Estimator for a TF-Ranking model.

```python
tfr.estimator.EstimatorBuilder(
    context_feature_columns, example_feature_columns, scoring_function,
    transform_function=None, optimizer=None, loss_reduction=None, hparams=None
)
```

<!-- Placeholder for "Used in" -->

An example use case is provided below:

```python
import tensorflow as tf
import tensorflow_ranking as tfr

def scoring_function(context_features, example_features, mode):
  # ...
  # scoring logic
  # ...
  return scores # tensors with shape [batch_size, ...]

context_feature_columns = {
  "c1": tf.feature_column.numeric_column("c1", shape=(1,))
}
example_feature_columns = {
  "e1": tf.feature_column.numeric_column("e1", shape=(1,))
}
hparams = dict(
    checkpoint_secs=120,
    loss="softmax_loss",
    model_dir="/path/to/your/model_dir/",
    num_checkpoints=100)
ranking_estimator = tfr.estimator.EstimatorBuilder(
      context_feature_columns,
      example_feature_columns,
      scoring_function=scoring_function,
      hparams=hparams).make_estimator()
```

If you want to customize certain `EstimatorBuilder` behaviors, please create a
subclass of `EstimatorBuilder`, and overwrite related functions. Right now, we
recommend only overwriting the `_eval_metric_fns` for your eval metrics. For
instance, if you need MAP (Mean Average Precision) as your evaluation metric,
you can do the following:

```python
class MyEstimatorBuilder(tfr.estimator.EstimatorBuilder):
  def _eval_metric_fns(self):
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.MAP, topn=topn) for topn in [5, 10]
    })
    return metric_fns

# Then, you can define your estimator with:
ranking_estimator = MyEstimatorBuilder(
      context_feature_columns,
      example_feature_columns,
      scoring_function=scoring_function,
      hparams=hparams).make_estimator()
```

If you really need to overwrite other functions, particularly `_transform_fn`,
`_group_score_fn` and `model_fn`, please be careful because the passed-in
parameters might no longer be used.

#### Args:

*   <b>`context_feature_columns`</b>: (dict) Context (aka, query) feature
    columns.
*   <b>`example_feature_columns`</b>: (dict) Example (aka, document) feature
    columns.
*   <b>`scoring_function`</b>: (function) A user-provided scoring function with
    the below signatures:
    *   Args: `context_features`: (dict) A dict of Tensors with the shape
        [batch_size, ...]. `example_features`: (dict) A dict of Tensors with the
        shape [batch_size, ...]. `mode`: (`estimator.ModeKeys`) Specifies if
        this is for training, evaluation or inference. See ModeKeys.
    *   Returns: The computed logits, a Tensor of shape [batch_size, 1].
*   <b>`transform_function`</b>: (function) A user-provided function that
    transforms raw features into dense Tensors with the following signature:
    *   Args: `features`: (dict) A dict of Tensors or SparseTensors containing
        the raw features from an `input_fn`. `mode`: (`estimator.ModeKeys`)
        Specifies if this is for training, evaluation or inference. See
        ModeKeys.
    *   Returns: `context_features`: (dict) A dict of Tensors with the shape
        [batch_size, ...]. `example_features`: (dict) A dict of Tensors with the
        shape [batch_size, list_size, ...].
*   <b>`optimizer`</b>: (`tf.Optimizer`) An `Optimizer` object for model
    optimzation.
*   <b>`loss_reduction`</b>: (str) An enum of strings indicating the loss
    reduction type. See type definition in the `tf.compat.v1.losses.Reduction`.
*   <b>`hparams`</b>: (dict) A dict containing model hyperparameters.

#### Raises:

*   <b>`ValueError`</b>: If the `example_feature_columns` is None.
*   <b>`ValueError`</b>: If the `scoring_function` is None..
*   <b>`ValueError`</b>: If both the `optimizer` and the
    `hparams["learning_rate"]` are not specified.

## Methods

<h3 id="make_estimator"><code>make_estimator</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/estimator.py">View
source</a>

```python
make_estimator()
```

Returns the built `tf.estimator.Estimator` for the TF-Ranking model.
