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

"""Tests for tfrbert_dataloader.py."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_ranking.python import data as tfr_data
from tensorflow_ranking.python.keras.tfrbert import tfrbert_dataloader as dataloader
from tensorflow_serving.apis import input_pb2


def _create_fake_preprocessed_dataset(output_path, seq_length, label_type):
  """Creates a fake dataset."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  elwc_num = 32
  list_size = 12
  for query_id in range(elwc_num):
    elwc = input_pb2.ExampleListWithContext()
    for doc_id in range(list_size):
      features = {}
      input_ids = np.random.randint(100, size=(seq_length))
      features['input_ids'] = create_int_feature(input_ids)
      features['input_mask'] = create_int_feature(np.ones_like(input_ids))
      features['segment_ids'] = create_int_feature(np.ones_like(input_ids))

      if label_type == 'int':
        features['relevance'] = create_int_feature([1])
      elif label_type == 'float':
        features['relevance'] = create_float_feature([0.5])
      else:
        raise ValueError('Unsupported label_type: %s' % label_type)

      features['query_id'] = create_int_feature([query_id])
      features['document_id'] = create_int_feature([doc_id])

      example = tf.train.Example(features=tf.train.Features(feature=features))
      elwc.examples.append(example)

    writer.write(elwc.SerializeToString())
  writer.close()


class TFRBERTDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('int', tf.int32), ('float', tf.float32))
  def test_load_dataset(self, label_type, expected_label_type):
    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    batch_size = 8
    list_size = 12
    seq_length = 128
    _create_fake_preprocessed_dataset(input_path, seq_length, label_type)
    data_config = dataloader.TFRBertDataConfig(
        input_path=input_path,
        data_format=tfr_data.ELWC,
        list_size=list_size,
        shuffle_examples=False,
        seq_length=seq_length,
        global_batch_size=batch_size,
        mask_feature_name='example_list_mask',
        label_feature_name='relevance',
        read_query_id=True,
        read_document_id=True,
        label_type=label_type)
    dataset = dataloader.TFRBertDataLoader(data_config).load()
    features, labels = next(iter(dataset))
    self.assertCountEqual(['input_word_ids', 'input_mask', 'input_type_ids',
                           'example_list_mask', 'query_id', 'document_id'],
                          features.keys())
    self.assertEqual(features['input_word_ids'].shape,
                     (batch_size, list_size, seq_length))
    self.assertEqual(features['input_mask'].shape,
                     (batch_size, list_size, seq_length))
    self.assertEqual(features['input_type_ids'].shape,
                     (batch_size, list_size, seq_length))
    self.assertEqual(features['example_list_mask'].shape,
                     (batch_size, list_size))
    self.assertEqual(features['query_id'].shape,
                     (batch_size, list_size, 1))
    self.assertEqual(features['document_id'].shape,
                     (batch_size, list_size, 1))

    self.assertEqual(labels.shape, (batch_size, list_size))
    self.assertEqual(labels.dtype, expected_label_type)


if __name__ == '__main__':
  tf.test.main()
