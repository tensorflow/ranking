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

# Lint as: python3
"""Adaptor between keras models and estimator."""

import tensorflow as tf

from tensorflow.python.util import function_utils
from tensorflow_ranking.python import utils
from tensorflow_ranking.python.keras import metrics


def model_to_estimator(model,
                       model_dir=None,
                       config=None,
                       custom_objects=None,
                       weights_feature_name=None,
                       warm_start_from=None):
  """Keras ranking model to Estimator.

  This function is based on the custom model_fn in TF2.0 migration guide.
  https://www.tensorflow.org/guide/migrate#custom_model_fn_with_tf_20_symbols

  Args:
    model: (tf.keras.Model) A ranking keras model, which  can be created using
      `tfr.keras.model.create_keras_model`. Masking is handled inside this
      function.
    model_dir: (str) Directory to save `Estimator` model graph and checkpoints.
    config: (tf.estimator.RunConfig) Specified config for distributed training
      and checkpointing.
    custom_objects: (dict) mapping names (strings) to custom objects (classes
      and functions) to be considered during deserialization.
    weights_feature_name: (str) A string specifying the name of the per-example
      (of shape [batch_size, list_size]) or per-list (of shape [batch_size, 1])
      weights feature in `features` dict.
    warm_start_from: (`tf.estimator.WarmStartSettings`) settings to warm-start
      the `tf.estimator.Estimator`.

  Returns:
    (tf.estimator.Estimator) A ranking estimator.

  Raises:
    ValueError: if weights_feature_name is not in features.
  """

  def _clone_fn(obj):
    """Clone keras object."""
    fn_args = function_utils.fn_args(obj.__class__.from_config)

    if "custom_objects" in fn_args:
      return obj.__class__.from_config(
          obj.get_config(), custom_objects=custom_objects)

    return obj.__class__.from_config(obj.get_config())

  def _model_fn(features, labels, mode, params, config):
    """Defines an `Estimator` `model_fn`."""
    del [config, params]

    # In Estimator, all sub-graphs need to be constructed inside the model_fn.
    # Hence, ranker, losses, metrics and optimizer are cloned inside this
    # function.
    ranker = tf.keras.models.clone_model(model, clone_function=_clone_fn)
    training = (mode == tf.compat.v1.estimator.ModeKeys.TRAIN)

    weights = None
    if weights_feature_name and mode != tf.compat.v1.estimator.ModeKeys.PREDICT:
      if weights_feature_name not in features:
        raise ValueError(
            "weights_feature '{0}' can not be found in 'features'.".format(
                weights_feature_name))
      else:
        weights = utils.reshape_to_2d(features.pop(weights_feature_name))

    logits = ranker(features, training=training)

    if mode == tf.compat.v1.estimator.ModeKeys.PREDICT:
      return tf.compat.v1.estimator.EstimatorSpec(mode=mode, predictions=logits)

    loss = _clone_fn(model.loss)
    total_loss = loss(labels, logits, sample_weight=weights)

    keras_metrics = []
    for metric in model.metrics:
      keras_metrics.append(_clone_fn(metric))
    # Adding default metrics here as model.metrics does not contain custom
    # metrics.
    keras_metrics += metrics.default_keras_metrics()
    eval_metric_ops = {}
    for keras_metric in keras_metrics:
      keras_metric.update_state(labels, logits, sample_weight=weights)
      eval_metric_ops[keras_metric.name] = keras_metric

    train_op = None
    if training:
      optimizer = _clone_fn(model.optimizer)
      optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
      # Get both the unconditional updates (the None part)
      # and the input-conditional updates (the features part).
      # These updates are for layers like BatchNormalization, which have
      # separate update and minimize ops.
      update_ops = ranker.get_updates_for(None) + ranker.get_updates_for(
          features)
      minimize_op = optimizer.get_updates(
          loss=total_loss, params=ranker.trainable_variables)[0]
      train_op = tf.group(minimize_op, *update_ops)

    return tf.compat.v1.estimator.EstimatorSpec(
        mode=mode,
        predictions=logits,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

  return tf.compat.v1.estimator.Estimator(
      model_fn=_model_fn,
      config=config,
      model_dir=model_dir,
      warm_start_from=warm_start_from)
