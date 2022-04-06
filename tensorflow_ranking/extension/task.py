# Copyright 2022 The TensorFlow Ranking Authors.
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

"""Orbit task for TF-Ranking.


Note: First - These APIs require These APS require the
`tensorflow_models`package. You can install it with `pip install
tf-models-official`. Second - Nothing under
`tensorflow_ranking.extension.premade` is imported by default. To use
these APIs import `premade` in your code:
`import tensorflow_ranking.extension.premade`.
"""
import dataclasses
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_ranking.python import data as tfr_data
from tensorflow_ranking.python.keras import losses as tfr_losses
from tensorflow_ranking.python.keras import metrics as tfr_metrics
from tensorflow_ranking.python.keras import model as tfr_model

# pylint: disable=g-import-not-at-top
try:
  from official.core import base_task
  from official.core import config_definitions as cfg
  from official.core import input_reader
  from official.core import task_factory
  from official.modeling import tf_utils
  from official.nlp.data import data_loader
except ModuleNotFoundError:
  raise ModuleNotFoundError(
      'tf-models-official needs to be installed. Run command: '
      '`pip install tf-models-official`.') from None
# pylint: enable=g-import-not-at-top

FeatureSpec = Dict[
    str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature, tf.io.RaggedFeature]]

DATASET_FN_MAP = {
    'tfrecord': tf.data.TFRecordDataset,
    'textline': tf.data.TextLineDataset
}

MASK = 'example_list_mask'
_PADDING_LABEL = -1.


def _convert_label(label, convert_labels_to_binary=False):
  """Converts the label to 1D and optionally binarizes it.

  Args:
    label: A tensor of shape [list_size, 1].
    convert_labels_to_binary: A boolean to indicate whether to use binary label.

  Returns:
    A tensor of shape [list_size].
  """
  if len(tf.shape(label)) > 1:
    label = tf.cast(tf.squeeze(label, axis=-1), tf.float32)
  if convert_labels_to_binary:
    label = tf.where(tf.greater(label, 0.), tf.ones_like(label), label)
  return label


@dataclasses.dataclass
class RankingDataConfig(cfg.DataConfig):
  """Data set config."""
  data_format: str = tfr_data.ELWC
  dataset_fn: str = 'tfrecord'
  list_size: Optional[int] = None
  shuffle_examples: bool = False
  convert_labels_to_binary: bool = False
  seed: Optional[int] = None
  mask_feature_name: Optional[str] = MASK
  is_training: bool = True
  drop_remainder: bool = True


class RankingDataLoader(data_loader.DataLoader):
  """A class to load dataset for ranking task."""

  def __init__(self,
               params,
               context_feature_spec: FeatureSpec = None,
               example_feature_spec: FeatureSpec = None,
               label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
               dataset_fn: Optional[Callable[[], tf.data.Dataset]] = None):
    self._params = params
    self._context_feature_spec = context_feature_spec or {}
    self._example_feature_spec = example_feature_spec or {}
    self._label_spec = label_spec
    self._dataset_fn = dataset_fn
    if not self._dataset_fn:
      if params.dataset_fn not in DATASET_FN_MAP:
        raise ValueError('Wrong dataset_fn: {}! Expected: {}'.format(
            params.dataset_fn, list(DATASET_FN_MAP.keys())))
      self._dataset_fn = DATASET_FN_MAP[params.dataset_fn]

  def _decode(self, record: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Decodes a serialized ELWC."""
    parsing_example_feature_spec = self._example_feature_spec
    if self._label_spec:
      parsing_example_feature_spec.update(dict([self._label_spec]))

    parsing_fn = tfr_data.make_parsing_fn(
        self._params.data_format,
        self._params.list_size,
        self._context_feature_spec,
        parsing_example_feature_spec,
        mask_feature_name=self._params.mask_feature_name,
        shuffle_examples=self._params.shuffle_examples,
        seed=self._params.seed)

    # The TF-Ranking parsing functions only takes batched ELWCs as input and
    # output a dictionary from feature names to Tensors with the shape of
    # (batch_size, list_size, feature_length).
    features = parsing_fn(tf.reshape(record, [1]))

    # Remove the first batch_size dimension and leave batching to DataLoader
    # class in construction of distributed data set.
    output_features = {
        name: tf.squeeze(tensor, 0) for name, tensor in features.items()
    }

    # ELWC only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in output_features:
      t = output_features[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      output_features[name] = t

    return output_features

  def _parse(
      self,
      record: Mapping[str,
                      tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    label_feature_name, _ = self._label_spec
    label = record.pop(label_feature_name)
    label = _convert_label(label, self._params.convert_labels_to_binary)

    return record, label

  def load(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params,
        dataset_fn=self._dataset_fn,
        decoder_fn=self._decode,
        parser_fn=self._parse)
    return reader.read(input_context)


@dataclasses.dataclass
class RankingTaskConfig(cfg.TaskConfig):
  """The TF-Ranking task config."""
  train_data: RankingDataConfig = None
  validation_data: RankingDataConfig = None
  loss: str = 'softmax_loss'
  # An enum of strings indicating the loss reduction type.
  # See type definition in the `tf.compat.v2.losses.Reduction`.
  # Only NONE works for TPU, while others may work on CPU/GPU.
  loss_reduction: str = tf.keras.losses.Reduction.NONE
  # If specified, group by `query_feature_name` to calculate aggregated metrics
  aggregated_metrics: bool = False
  # If True, output prediction results to a csv after each validation step.
  output_preds: bool = False


@task_factory.register_task_cls(RankingTaskConfig)
class RankingTask(base_task.Task):
  """Task object for TF-Ranking."""

  def __init__(self,
               params,
               model_builder: tfr_model.AbstractModelBuilder,
               context_feature_spec: FeatureSpec = None,
               example_feature_spec: FeatureSpec = None,
               label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
               dataset_fn: Optional[Callable[[], tf.data.Dataset]] = None,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None):
    super().__init__(params=params,
                     logging_dir=logging_dir,
                     name=name)
    self._model_builder = model_builder
    self._context_feature_spec = context_feature_spec or {}
    self._example_feature_spec = example_feature_spec or {}
    self._label_spec = label_spec
    self._dataset_fn = dataset_fn

  def build_model(self):
    return self._model_builder.build()

  def build_inputs(self, params, input_context=None):
    ranking_dataloader = RankingDataLoader(
        params,
        context_feature_spec=self._context_feature_spec,
        example_feature_spec=self._example_feature_spec,
        label_spec=self._label_spec,
        dataset_fn=self._dataset_fn)
    return ranking_dataloader.load(input_context)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    ranking_loss = tfr_losses.get(
        loss=self.task_config.loss, reduction=self.task_config.loss_reduction)
    loss = ranking_loss(tf.cast(labels, tf.float32),
                        tf.cast(model_outputs, tf.float32))
    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf_utils.safe_mean(loss)

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
    return logs
