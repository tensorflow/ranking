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

"""Loads dataset for TFR-BERT."""
from typing import Dict, Mapping, Optional, Tuple

import dataclasses
import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from tensorflow_ranking.python import data as tfr_data

_PADDING_LABEL = -1
LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}
DATASET_FN_MAP = {
    'tfrecord': tf.data.TFRecordDataset,
    'textline': tf.data.TextLineDataset
}
MASK = 'example_list_mask'
QUERY_ID = 'query_id'
DOCUMENT_ID = 'document_id'


@dataclasses.dataclass
class TFRBertDataConfig(cfg.DataConfig):
  """Data config for TFR-BERT task."""
  # TFR-BERT configs
  seq_length: int = 128
  label_feature_name: str = 'label'
  read_query_id: bool = False  # Only supports int64 ids.
  read_document_id: bool = False  # Only supports int64 ids.
  # TF-Ranking configs
  data_format: str = tfr_data.ELWC
  dataset_fn: str = 'tfrecord'
  list_size: Optional[int] = None
  shuffle_examples: bool = False
  label_type: str = 'float'
  seed: Optional[int] = None
  mask_feature_name: Optional[str] = MASK
  is_training: bool = True
  drop_remainder: bool = True


class TFRBertDataLoader(data_loader.DataLoader):
  """A class to load dataset for TFR-BERT task."""

  def __init__(self, params):
    self._params = params
    self._extra_feature_names = []
    if params.read_query_id:
      self._extra_feature_names.append(QUERY_ID)
    if params.read_document_id:
      self._extra_feature_names.append(DOCUMENT_ID)
    if params.dataset_fn not in DATASET_FN_MAP:
      raise ValueError('Wrong dataset_fn: {}! Expected: {}'.format(
          params.dataset_fn, list(DATASET_FN_MAP.keys())))
    self._dataset_fn = DATASET_FN_MAP[params.dataset_fn]

  def _decode(self, record: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Decodes a serialized ELWC."""
    label_type = LABEL_TYPES_MAP[self._params.label_type]

    context_feature_spec = {}
    example_feature_spec = {
        'input_ids': tf.io.FixedLenFeature(
            shape=(self._params.seq_length,), dtype=tf.int64,
            default_value=[0] * self._params.seq_length),
        'input_mask': tf.io.FixedLenFeature(
            shape=(self._params.seq_length,), dtype=tf.int64,
            default_value=[0] * self._params.seq_length),
        'segment_ids': tf.io.FixedLenFeature(
            shape=(self._params.seq_length,), dtype=tf.int64,
            default_value=[0] * self._params.seq_length),
        self._params.label_feature_name: tf.io.FixedLenFeature(
            shape=[], dtype=label_type, default_value=_PADDING_LABEL)}
    if self._extra_feature_names:
      example_feature_spec.update({
          extra_feature_name: tf.io.FixedLenFeature(
              shape=(1,), dtype=tf.int64, default_value=-1)
          for extra_feature_name in self._extra_feature_names
      })

    parsing_fn = tfr_data.make_parsing_fn(
        self._params.data_format,
        self._params.list_size,
        context_feature_spec,
        example_feature_spec,
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
        name: tf.squeeze(tensor, 0)
        for name, tensor in features.items()
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
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    if self._params.mask_feature_name in record:
      x.update(
          {self._params.mask_feature_name:
               record[self._params.mask_feature_name]})
    if self._extra_feature_names:
      x.update({f: record[f] for f in self._extra_feature_names})

    y = record[self._params.label_feature_name]
    return (x, y)

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
