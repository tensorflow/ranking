# Copyright 2020 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides an `EstimatorBuilder` for creating a TF-Ranking model estimator.

This class contains the boilerplate that is required to create an estimator for
a TF-Ranking model. The goal is to reduce replicated setups (e.g., transform
function, scoring function) for adopting TF-Ranking. Advanced users can also
derive from this class and further tailor for their needs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf

from tensorflow_ranking.python import feature
from tensorflow_ranking.python import head
from tensorflow_ranking.python import losses
from tensorflow_ranking.python import metrics
from tensorflow_ranking.python import model


def _validate_hparams(hparams_in_dict, required_keys):
  """Asserts all of the `required_keys` are presented in `hparams_in_dict`.

  Args:
    hparams_in_dict: (dict) A dict with the key in string and value in any type.
    required_keys: (list) A list of strings.

  Raises:
    ValueError: If any of `required_keys` does not present in `hparams_in_dict`.
  """
  for required_key in required_keys:
    if required_key not in hparams_in_dict:
      raise ValueError("Required key is missing: '{}'".format(required_key))


def _validate_function_args(function, required_args):
  """Asserts all of the `required_args` are presented in `function` args.

  Args:
    function: (function) A python function.
    required_args: (list) A list of strings indicating the required args.

  Raises:
    ValueError: If any of `required_args` does not present in the `function`.
  """
  fn_args = None
  try:
    # This is for Python 3.
    fn_spec = inspect.getfullargspec(function)
    fn_args = [arg for arg in fn_spec.args if arg != "self"]
  except AttributeError:
    # This is for Python 2.
    fn_spec = inspect.getargspec(function)
    fn_args = [arg for arg in fn_spec.args if arg != "self"]

  if set(fn_args) != set(required_args):
    raise ValueError(
        "Function `%s` needs to have the following arguments: %s."
        " What were provided are the following: %s." %
        (function.__name__, sorted(required_args), sorted(fn_args)))


class EstimatorBuilder(object):
  """Builds a tf.estimator.Estimator for a TF-Ranking model.

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
      listwise_inference=False,
      loss="softmax_loss",
      model_dir="/path/to/your/model_dir/",
      num_checkpoints=100)
  ranking_estimator = tfr.estimator.EstimatorBuilder(
        context_feature_columns,
        example_feature_columns,
        scoring_function=scoring_function,
        hparams=hparams).make_estimator()
  ```

  If you want to customize certain `EstimatorBuilder` behaviors, please create
  a subclass of `EstimatorBuilder`, and overwrite related functions. Right now,
  we recommend only overwriting the `_eval_metric_fns` for your eval metrics.
  For instance, if you need MAP (Mean Average Precision) as your evaluation
  metric, you can do the following:

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
  """

  def __init__(self,
               context_feature_columns,
               example_feature_columns,
               scoring_function,
               transform_function=None,
               optimizer=None,
               loss_reduction=None,
               hparams=None):
    """Constructor.

    Args:
      context_feature_columns: (dict) Context (aka, query) feature columns.
      example_feature_columns: (dict) Example (aka, document) feature columns.
      scoring_function: (function) A user-provided scoring function with the
        below signatures:
        * Args:
          `context_features`: (dict) A dict of Tensors with the shape
            [batch_size, ...].
          `example_features`: (dict) A dict of Tensors with the shape
            [batch_size, ...].
          `mode`: (`estimator.ModeKeys`) Specifies if this is for training,
            evaluation or inference. See ModeKeys.
        * Returns: The computed logits, a Tensor of shape [batch_size, 1].
      transform_function: (function) A user-provided function that transforms
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
      optimizer: (`tf.Optimizer`) An `Optimizer` object for model optimzation.
      loss_reduction: (str) An enum of strings indicating the loss reduction
        type. See type definition in the `tf.compat.v1.losses.Reduction`.
      hparams: (dict) A dict containing model hyperparameters.

    Raises:
      ValueError: If the `example_feature_columns` is None.
      ValueError: If the `scoring_function` is None..
      ValueError: If both the `optimizer` and the `hparams["learning_rate"]`
        are not specified.
    """
    if example_feature_columns is None:
      raise ValueError("The `example_feature_columns` is not specified!")

    if scoring_function is None:
      raise ValueError("The `scoring_function` needs to be specified!")

    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    self._scoring_function = scoring_function
    self._transform_function = transform_function
    self._hparams = hparams
    self._validate_function_args_and_hparams()

    if not optimizer and not hparams.get("learning_rate"):
      raise ValueError("Please specify either the `optimizer` or the "
                       "`learning_rate` in `hparams`!")
    if optimizer and hparams.get("learning_rate"):
      tf.compat.v1.logging.warning("`learning_rate` from `hparams` is ignored "
                                   "as the `optimizer` has been specified!")
    self._optimizer = (
        optimizer or tf.compat.v1.train.AdagradOptimizer(
            learning_rate=hparams.get("learning_rate")))

    self._loss_reduction = loss_reduction or tf.compat.v1.losses.Reduction.SUM

  def _required_hparam_keys(self):
    """Returns a list of keys for required hparams."""
    required_hparam_keys = [
        "checkpoint_secs", "listwise_inference", "loss", "model_dir",
        "num_checkpoints"
    ]
    return required_hparam_keys

  def _validate_function_args_and_hparams(self):
    """Validates that the hparams and arguments are all as required."""
    _validate_hparams(self._hparams, self._required_hparam_keys())
    _validate_function_args(
        self._scoring_function,
        required_args=["context_features", "example_features", "mode"])
    if self._transform_function is not None:
      _validate_function_args(
          self._transform_function, required_args=["features", "mode"])

  def _transform_fn(self, features, mode):
    """Defines the transform fn."""
    if self._transform_function is not None:
      return self._transform_function(features=features, mode=mode)

    if (mode == tf.estimator.ModeKeys.PREDICT and
        not self._hparams.get("listwise_inference")):
      return feature.encode_pointwise_features(
          features=features,
          context_feature_columns=self._context_feature_columns,
          example_feature_columns=self._example_feature_columns,
          mode=mode,
          scope="transform_layer")
    else:
      return feature.encode_listwise_features(
          features=features,
          context_feature_columns=self._context_feature_columns,
          example_feature_columns=self._example_feature_columns,
          mode=mode,
          scope="transform_layer")

  def _eval_metric_fns(self):
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg_%d" % topn: metrics.make_ranking_metric_fn(
            metrics.RankingMetricKey.NDCG, topn=topn) for topn in [5, 10]
    })
    metric_fns.update({
        "metric/mrr_%d" % topn:
        metrics.make_ranking_metric_fn(metrics.RankingMetricKey.MRR, topn=topn)
        for topn in [10]
    })
    metric_fns.update({
        "metric/%s" % name: metrics.make_ranking_metric_fn(name) for name in
        [metrics.RankingMetricKey.MRR, metrics.RankingMetricKey.NDCG]
    })
    return metric_fns

  def _group_score_fn(self, context_features, group_features, mode, params,
                      config):
    """Returns a groupwise score fn to build `EstimatorSpec`."""
    del [params, config]  # They are not used.

    # Squeeze the group features because we are in univariate mode.
    example_features = {}
    for k, v in group_features.items():
      example_features[k] = tf.squeeze(v, 1)

    if self._scoring_function is None:
      raise ValueError("The attribute `scoring_function` is being used before"
                       "being assigned.")
    return self._scoring_function(
        context_features=context_features,
        example_features=example_features,
        mode=mode)

  def _model_fn(self):
    """Returns a model_fn."""

    def _train_op_fn(loss):
      """Defines train op used in ranking head."""
      update_ops = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.UPDATE_OPS)
      minimize_op = self._optimizer.minimize(
          loss=loss, global_step=tf.compat.v1.train.get_global_step())
      train_op = tf.group([update_ops, minimize_op])
      return train_op

    ranking_head = head.create_ranking_head(
        loss_fn=losses.make_loss_fn(
            self._hparams.get("loss"), reduction=self._loss_reduction),
        eval_metric_fns=self._eval_metric_fns(),
        train_op_fn=_train_op_fn)

    return model.make_groupwise_ranking_fn(
        group_score_fn=self._group_score_fn,
        group_size=1,
        transform_fn=self._transform_fn,
        ranking_head=ranking_head)

  def make_estimator(self):
    """Returns the built `tf.estimator.Estimator` for the TF-Ranking model."""
    config = tf.estimator.RunConfig(
        model_dir=self._hparams.get("model_dir"),
        keep_checkpoint_max=self._hparams.get("num_checkpoints"),
        save_checkpoints_secs=self._hparams.get("checkpoint_secs"))
    return tf.estimator.Estimator(model_fn=self._model_fn(), config=config)


def _make_dnn_score_fn(hidden_units,
                       activation_fn=tf.nn.relu,
                       dropout=None,
                       use_batch_norm=False,
                       batch_norm_moment=0.999):
  """Returns a DNN scoring fn that outputs a score per example.

  Args:
    hidden_units: (list) Iterable of number hidden units per layer for a DNN
      model. All layers are fully connected. Ex. `[64, 32]` means first layer
      has 64 nodes and second one has 32.
    activation_fn: Activation function applied to each layer. If `None`, will
      use `tf.nn.relu`.
    dropout: (float) When not `None`, the probability we will drop out a given
      coordinate.
    use_batch_norm: (bool) If true, use batch normalization after each hidden
      layer.
    batch_norm_moment: (float) Momentum for the moving average in batch
      normalization.

  Returns:
    A DNN scoring function.
  """
  activation_fn = activation_fn or tf.nn.relu

  def _scoring_function(context_features, example_features, mode):
    """Defines the DNN-based scoring fn.

    Args:
      context_features: (dict) A mapping from context feature names to dense 2-D
        Tensors of shape [batch_size, ...].
      example_features: (dict) A mapping from example feature names to dense 3-D
        Tensors of shape [batch_size, list_size, ...].
      mode: (`tf.estimator.ModeKeys`) TRAIN, EVAL, or PREDICT.

    Returns:
      A Tensor of shape [batch_size, 1] containing per-example.
      scores.
    """
    # Input layer.
    with tf.compat.v1.name_scope("input_layer"):
      example_input = [
          tf.compat.v1.layers.flatten(example_features[name])
          for name in sorted(list(example_features.keys()))
      ]
      context_input = [
          tf.compat.v1.layers.flatten(context_features[name])
          for name in sorted(list(context_features.keys()))
      ]
      # Concat context and example features as input.
      input_layer = tf.concat(context_input + example_input, 1)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = input_layer
    # Construct a deep neural network model.
    with tf.compat.v1.name_scope("dnn_model"):
      if use_batch_norm:
        cur_layer = tf.compat.v1.layers.batch_normalization(
            cur_layer, training=is_training, momentum=batch_norm_moment)
      for i, layer_width in enumerate(map(int, hidden_units)):
        cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
        if use_batch_norm:
          cur_layer = tf.compat.v1.layers.batch_normalization(
              cur_layer, training=is_training, momentum=batch_norm_moment)
        cur_layer = activation_fn(cur_layer)
        if dropout:
          cur_layer = tf.compat.v1.layers.dropout(
              inputs=cur_layer, rate=dropout, training=is_training)
        tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                    tf.nn.zero_fraction(cur_layer))
      logits = tf.compat.v1.layers.dense(cur_layer, units=1)

    tf.compat.v1.summary.scalar("logits_mean",
                                tf.reduce_mean(input_tensor=logits))
    return logits

  return _scoring_function


def make_dnn_ranking_estimator(
    example_feature_columns,
    hidden_units,
    context_feature_columns=None,
    optimizer=None,
    learning_rate=0.05,
    listwise_inference=False,
    loss="approx_ndcg_loss",
    loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
    activation_fn=tf.nn.relu,
    dropout=None,
    use_batch_norm=False,
    batch_norm_moment=0.999,
    model_dir=None,
    checkpoint_secs=120,
    num_checkpoints=1000):
  """Builds an `Estimator` instance with DNN scoring function.

  Args:
    example_feature_columns: (dict) Example (aka, document) feature columns.
    hidden_units: (list) Iterable of number hidden units per layer for a DNN
      model. All layers are fully connected. Ex. `[64, 32]` means first layer
      has 64 nodes and second one has 32.
    context_feature_columns: (dict) Context (aka, query) feature columns.
    optimizer: (`tf.Optimizer`) An `Optimizer` object for model optimzation.
    learning_rate: (float) Only used if `optimizer` is a string. Defaults to
      0.05.
    listwise_inference: (bool) Whether the inference will be performed with the
      listwise data format such as `ExampleListWithContext`.
    loss: (str) A string to decide the loss function used in training. See
      `RankingLossKey` class for possible values.
    loss_reduction: (str) An enum of strings indicating the loss reduction type.
      See type definition in the `tf.compat.v1.losses.Reduction`.
    activation_fn: Activation function applied to each layer. If `None`, will
      use `tf.nn.relu`.
    dropout: (float) When not `None`, the probability we will drop out a given
      coordinate.
    use_batch_norm: (bool) Whether to use batch normalization after each hidden
      layer.
    batch_norm_moment: (float) Momentum for the moving average in batch
      normalization.
    model_dir: (str) Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    checkpoint_secs: (int) Time interval (in seconds) to save checkpoints.
    num_checkpoints: (int) Number of checkpoints to keep.

  Returns:
    An `Estimator` with DNN scoring function.
  """

  scoring_function = _make_dnn_score_fn(
      hidden_units,
      activation_fn=activation_fn,
      dropout=dropout,
      use_batch_norm=use_batch_norm,
      batch_norm_moment=batch_norm_moment)

  hparams = dict(
      model_dir=model_dir,
      learning_rate=learning_rate,
      listwise_inference=listwise_inference,
      loss=loss,
      checkpoint_secs=checkpoint_secs,
      num_checkpoints=num_checkpoints)

  return EstimatorBuilder(
      context_feature_columns,
      example_feature_columns,
      optimizer=optimizer,
      scoring_function=scoring_function,
      loss_reduction=loss_reduction,
      hparams=hparams).make_estimator()
