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

r"""Tests for tfrbert_task.py."""

import os

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.nlp.configs import encoders
from tensorflow_ranking.extension.premade import tfrbert_task
from tensorflow_ranking.python import data as tfr_data
from tensorflow_ranking.python.keras import model as tfr_model
from tensorflow_serving.apis import input_pb2


FLAGS = flags.FLAGS


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

      if label_type == tf.int64:
        features['relevance'] = create_int_feature([1])
      elif label_type == tf.float32:
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

  @parameterized.parameters((tf.int64,), (tf.float32,))
  def test_load_dataset(self, label_type):
    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    batch_size = 8
    list_size = 12
    seq_length = 128
    _create_fake_preprocessed_dataset(input_path, seq_length, label_type)
    label_spec = ('relevance', tf.io.FixedLenFeature(
        shape=[1,], dtype=label_type, default_value=-1))
    data_config = tfrbert_task.TFRBertDataConfig(
        input_path=input_path,
        data_format=tfr_data.ELWC,
        list_size=list_size,
        shuffle_examples=False,
        seq_length=seq_length,
        global_batch_size=batch_size,
        mask_feature_name='example_list_mask',
        read_query_id=True,
        read_document_id=True)
    dataset = tfrbert_task.TFRBertDataLoader(data_config, label_spec).load()
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


class ModelBuilderTest(tf.test.TestCase):

  def _create_input_data(self):
    dummy_word_ids = tf.random.uniform(
        minval=0, maxval=100, shape=(12, 10, 128), dtype=tf.int64)
    dummy_mask = tf.ones((12, 10, 128))
    dummy_type_ids = tf.zeros((12, 10, 128))
    dummy_example_list_mask = tf.ones((12, 10), dtype=tf.bool)
    x = dict(
        input_word_ids=dummy_word_ids,
        input_mask=dummy_mask,
        input_type_ids=dummy_type_ids,
        example_list_mask=dummy_example_list_mask)
    return x

  def test_tfr_bert_model_builder(self):
    encoder_config = encoders.EncoderConfig(
        bert=encoders.BertEncoderConfig(vocab_size=30522, num_layers=1))
    encoder_network = encoders.build_encoder(encoder_config)
    preprocess_dict = {}
    scorer = tfrbert_task.TFRBertScorer(
        encoder=encoder_network,
        bert_output_dropout=0.1)

    example_feature_spec = {
        'input_word_ids': tf.io.FixedLenFeature(shape=(None,), dtype=tf.int64),
        'input_mask': tf.io.FixedLenFeature(shape=(None,), dtype=tf.int64),
        'input_type_ids': tf.io.FixedLenFeature(shape=(None,), dtype=tf.int64)
    }
    context_feature_spec = {}

    model_builder = tfrbert_task.TFRBertModelBuilder(
        input_creator=tfr_model.FeatureSpecInputCreator(
            context_feature_spec, example_feature_spec),
        preprocessor=tfr_model.PreprocessorWithSpec(preprocess_dict),
        scorer=scorer,
        mask_feature_name='example_list_mask',
        name='tfrbert_model')
    model = model_builder.build()

    output = model(self._create_input_data())
    self.assertAllEqual(output.shape.as_list(), [12, 10])


class TFRBertTaskTest(tf.test.TestCase):

  def setUp(self):
    super(TFRBertTaskTest, self).setUp()
    self._logging_dir = self.get_temp_dir()

  def _create_bert_ckpt(self):
    config = encoders.EncoderConfig(
        type='bert', bert=encoders.BertEncoderConfig(num_layers=1))
    encoder = encoders.build_encoder(config)
    ckpt = tf.train.Checkpoint(encoder=encoder)
    ckpt_path = ckpt.save(os.path.join(self._logging_dir, 'ckpt'))
    return ckpt_path

  def test_task(self):
    # Prepare check point and test data
    ckpt_path = self._create_bert_ckpt()

    input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    seq_length = 128
    _create_fake_preprocessed_dataset(input_path, seq_length, tf.float32)

    # Set up data config
    train_data_config = tfrbert_task.TFRBertDataConfig(
        input_path=input_path,
        is_training=True,
        global_batch_size=5,
        list_size=3,
        dataset_fn='tfrecord',
        seq_length=128)
    validation_data_config = tfrbert_task.TFRBertDataConfig(
        input_path=input_path,
        is_training=False,
        global_batch_size=5,
        list_size=3,
        dataset_fn='tfrecord',
        seq_length=128,
        read_query_id=True,
        read_document_id=True)

    # Set up task config
    task_config = tfrbert_task.TFRBertConfig(
        output_preds=True,
        init_checkpoint=ckpt_path,
        aggregated_metrics=True,
        train_data=train_data_config,
        validation_data=validation_data_config,
        model=tfrbert_task.TFRBertModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(num_layers=1))))

    # Set up TFRBertTask
    label_spec = ('label',
                  tf.io.FixedLenFeature(
                      shape=(1,), dtype=tf.int64, default_value=-1))
    task = tfrbert_task.TFRBertTask(
        task_config,
        label_spec=label_spec,
        dataset_fn=tf.data.TFRecordDataset,
        logging_dir=self._logging_dir)

    # Test
    model = task.build_model()
    metrics = task.build_metrics()
    train_dataset = task.build_inputs(task_config.train_data)
    vali_dataset = task.build_inputs(task_config.validation_data)
    task.initialize(model)
    train_iterator = iter(train_dataset)
    vali_iterator = iter(vali_dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(train_iterator), model, optimizer, metrics=metrics)
    logs = task.validation_step(next(vali_iterator), model, metrics=metrics)
    logs = {x: (logs[x],) for x in logs}
    logs = task.aggregate_logs(step_outputs=logs)
    self.assertEqual(tf.constant(logs['query_id']).shape, (1, 5, 3))
    self.assertEqual(tf.constant(logs['document_id']).shape, (1, 5, 3))
    self.assertEqual(
        tf.constant(logs[tfrbert_task._PREDICTION]).shape, (1, 5, 3))
    self.assertEqual(tf.constant(logs[tfrbert_task._LABEL]).shape, (1, 5, 3))
    metrics = task.reduce_aggregated_logs(logs)


if __name__ == '__main__':
  tf.test.main()
