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


def model_to_estimator(model, model_dir=None, config=None):
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

  Returns:
    (tf.estimator.Estimator) A ranking estimator.
  """

  clone_keras_obj = lambda obj: obj.__class__.from_config(obj.get_config())

  def _model_fn(features, labels, mode, params, config):
    """Defines an `Estimator` `model_fn`."""
    del [config, params]

    # In Estimator, all sub-graphs need to be constructed inside the model_fn.
    # Hence, ranker, losses, metrics and optimizer are cloned inside this
    # function.
    ranker = tf.keras.models.clone_model(model)
    training = (mode == tf.compat.v1.estimator.ModeKeys.TRAIN)
    logits = ranker(features, training=training)

    if mode == tf.compat.v1.estimator.ModeKeys.PREDICT:
      return tf.compat.v1.estimator.EstimatorSpec(mode=mode, predictions=logits)

    loss = clone_keras_obj(model.loss)
    total_loss = loss(labels, logits)

    keras_metrics = []
    for metric in model.metrics:
      keras_metrics.append(clone_keras_obj(metric))
    eval_metric_ops = {}
    for keras_metric in keras_metrics:
      keras_metric.update_state(labels, logits)
      eval_metric_ops[keras_metric.name] = keras_metric

    train_op = None
    if training:
      optimizer = clone_keras_obj(model.optimizer)
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
      model_fn=_model_fn, config=config, model_dir=model_dir)
