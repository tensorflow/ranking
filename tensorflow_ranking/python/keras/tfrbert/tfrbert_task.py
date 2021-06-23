# Copyright 2021 The TensorFlow Ranking Authors.
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

"""TF-Ranking BERT task."""
import os
import time

from absl import logging
import dataclasses
import numpy as np
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from tensorflow_ranking.python.keras import losses as tfr_losses
from tensorflow_ranking.python.keras import metrics as tfr_metrics
from tensorflow_ranking.python.keras import model as tfr_model
from tensorflow_ranking.python.keras.tfrbert import tfrbert_dataloader as dataloader
from tensorflow_ranking.python.keras.tfrbert import tfrbert_model

_PADDING_LABEL = -1.
_PREDICTION = 'prediction'
_LABEL = 'label'


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A TFR-BERT model configuration."""
  dropout_rate: float = 0.1
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()


@dataclasses.dataclass
class TFRBertConfig(cfg.TaskConfig):
  """The tf-ranking BERT task config."""
  init_checkpoint: str = ''
  loss: str = 'softmax_loss'
  # An enum of strings indicating the loss reduction type.
  # See type definition in the `tf.compat.v2.losses.Reduction`.
  # Only NONE works for TPU, while others may work on CPU/GPU.
  loss_reduction: str = tf.keras.losses.Reduction.NONE
  # Defines the concrete model config at instantiation time.
  model: ModelConfig = ModelConfig()
  train_data: dataloader.TFRBertDataConfig = dataloader.TFRBertDataConfig()
  validation_data: dataloader.TFRBertDataConfig = dataloader.TFRBertDataConfig()
  # If specified, group by `query_feature_name` to calculate aggregated metrics
  aggregated_metrics: bool = False
  # If True, output prediction results to a csv after each validation step.
  output_preds: bool = False


@task_factory.register_task_cls(TFRBertConfig)
class TFRBertTask(base_task.Task):
  """Task object for tf-ranking BERT."""

  def build_model(self):
    encoder_network = encoders.build_encoder(self.task_config.model.encoder)
    preprocess_dict = {}
    scorer = tfrbert_model.TFRBertScorer(
        encoder=encoder_network,
        bert_output_dropout=self.task_config.model.dropout_rate)

    example_feature_spec = {
        'input_word_ids': tf.io.FixedLenFeature(
            shape=(None,), dtype=tf.int64),
        'input_mask': tf.io.FixedLenFeature(
            shape=(None,), dtype=tf.int64),
        'input_type_ids': tf.io.FixedLenFeature(
            shape=(None,), dtype=tf.int64)}
    context_feature_spec = {}

    model_builder = tfrbert_model.TFRBertModelBuilder(
        input_creator=tfr_model.FeatureSpecInputCreator(
            context_feature_spec, example_feature_spec),
        preprocessor=tfr_model.PreprocessorWithSpec(preprocess_dict),
        scorer=scorer,
        mask_feature_name=dataloader.MASK,
        name='tfrbert_model')
    return model_builder.build()

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    ranking_loss = tfr_losses.get(
        loss=self.task_config.loss, reduction=self.task_config.loss_reduction)
    loss = ranking_loss(tf.cast(labels, tf.float32),
                        tf.cast(model_outputs, tf.float32))
    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf_utils.safe_mean(loss)

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for tf-ranking BERT task."""
    return dataloader.TFRBertDataLoader(params).load(input_context)

  def build_metrics(self, training=None):
    del training
    metrics = [
        tfr_metrics.MeanAveragePrecisionMetric(name='MAP')
    ]
    for topn in [1, 5, 10]:
      metrics.append(
          tfr_metrics.NDCGMetric(name='NDCG@{}'.format(topn), topn=topn))
    for topn in [1, 5, 10]:
      metrics.append(
          tfr_metrics.MRRMetric(name='MRR@{}'.format(topn), topn=topn))
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    for metric in metrics:
      metric.update_state(labels, model_outputs)

  def train_step(self, inputs, model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer, metrics):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      features, labels = inputs
    else:
      features, labels = inputs, inputs
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Computes per-replica loss.
      loss = self.build_losses(
          labels=labels, model_outputs=outputs, aux_losses=model.losses)
      scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    self.process_metrics(metrics, labels, outputs)
    return {self.loss: loss}

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      features, labels = inputs
    else:
      features, labels = inputs, inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses)
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    logs.update({
        _PREDICTION: outputs,
        _LABEL: labels
    })
    # Collect extra feature values like query ids and document ids.
    # Notice that only 1-d features are supported.
    logging_feature_names = self._get_logging_feature_names()
    logs.update({
        feature_name: tf.squeeze(features[feature_name], axis=list(range(
            2, len(tf.shape(features[feature_name])))))
        for feature_name in logging_feature_names
    })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    """Aggregates over logs. This runs on CPU in eager mode."""
    logging_keys = list(self._get_logging_feature_names())
    logging_keys.extend([_PREDICTION, _LABEL])
    if state is None:
      state = {key: [] for key in logging_keys}
    for key in logging_keys:
      state[key].append(
          np.concatenate([v.numpy() for v in step_outputs[key]], axis=0))
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    """Calculates aggregated metrics and writes predictions to csv."""
    # Aggregate and flatten all the logs.
    flattened_aggregated_logs = {
        key: np.reshape(np.concatenate(logs, axis=0), -1)
        for key, logs in aggregated_logs.items()
    }

    # Output predictions, labels, query ids and document ids to a csv file.
    if self.task_config.output_preds:
      output_dir = os.path.join(self._logging_dir, 'prediction')
      if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
      flattened_output_logs = {
          key: flattened_aggregated_logs[key]
          for key in list(
              self._get_output_feature_names()) + [_PREDICTION, _LABEL]
      }
      output_path = os.path.join(
          output_dir, '%s.csv' % (time.strftime('%s', time.localtime())))
      self._write_as_csv(flattened_output_logs, output_path)

    # Calculate aggregated metrics.
    output = {}
    if self.task_config.aggregated_metrics:
      output = self._calculate_aggregated_metrics(
          flattened_aggregated_logs, dataloader.QUERY_ID)

    return output

  def _get_output_feature_names(self):
    """Returns a set of feature names that needs to be output."""
    feature_names = set()
    if self.task_config.output_preds:
      feature_names = feature_names.union(
          {dataloader.QUERY_ID, dataloader.DOCUMENT_ID})
    return feature_names

  def _get_logging_feature_names(self):
    """Returns a set of feature names that needs to be logged."""
    feature_names = self._get_output_feature_names()
    if self.task_config.aggregated_metrics:
      feature_names.add(dataloader.QUERY_ID)
    return feature_names

  def _calculate_aggregated_metrics(
      self, flattened_aggregated_logs, query_feature_name):
    """Calculates metrics where lists are grouped by `query_feature_name`."""
    qid2labels = {}
    qid2preds = {}

    qids = flattened_aggregated_logs[query_feature_name]
    preds = flattened_aggregated_logs[_PREDICTION]
    labels = flattened_aggregated_logs[_LABEL]
    for qid, pred, label in zip(qids, preds, labels):
      qid2labels[qid] = qid2labels.get(qid, []) + [label]
      qid2preds[qid] = qid2preds.get(qid, []) + [pred]

    metrics = [
        tfr_metrics.MeanAveragePrecisionMetric(name='Aggregated_MAP')
    ]
    for topn in [1, 5, 10]:
      metrics.append(
          tfr_metrics.NDCGMetric(
              name='Aggregated_NDCG@{}'.format(topn), topn=topn))
    for topn in [1, 5, 10]:
      metrics.append(
          tfr_metrics.MRRMetric(
              name='Aggregated_MRR@{}'.format(topn), topn=topn))

    output_results = {}
    for metric in metrics:
      for qid in qid2preds:
        preds = np.expand_dims(qid2preds[qid], 0)
        labels = np.expand_dims(qid2labels[qid], 0)
        metric.update_state(labels, preds)
      output_results.update({
          'agggregated_metrics/{}'.format(metric.name):
              metric.result().numpy()})
      logging.info('agggregated_metrics/%s = %f',
                   metric.name, metric.result().numpy())
    return output_results

  def _write_as_csv(self, outputs, output_path):
    """Writes prediction results to a csv file."""
    # Using `sorted_keys` to make sure results are properly aligned.
    logging.info('Writing prediction results to %s', output_path)
    sorted_keys = sorted(list(outputs.keys()))
    csv_outputs = []
    for i in range(len(next(iter(outputs.values())))):
      csv_output = [str(outputs[key][i]) for key in sorted_keys]
      csv_outputs.append(','.join(csv_output))

    with tf.io.gfile.GFile(output_path, 'w') as writer:
      writer.write(','.join(sorted_keys) + '\n')
      writer.write('\n'.join(csv_outputs))

  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    pretrain2finetune_mapping = {
        'encoder': model.checkpoint_items['encoder'],
    }
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)
