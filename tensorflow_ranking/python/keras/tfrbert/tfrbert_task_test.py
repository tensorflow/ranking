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

r"""Tests for tfrbert_task.py."""

import os
from absl import flags
import tensorflow as tf

from official.nlp.configs import encoders
from tensorflow_ranking.python.keras.tfrbert import tfrbert_task

FLAGS = flags.FLAGS


class TFRBertTaskTest(tf.test.TestCase):

  def setUp(self):
    super(TFRBertTaskTest, self).setUp()
    self._logging_dir = self.get_temp_dir()

  def _create_bert_ckpt(self):
    config = encoders.EncoderConfig(
        type="bert", bert=encoders.BertEncoderConfig(num_layers=1))
    encoder = encoders.build_encoder(config)
    ckpt = tf.train.Checkpoint(encoder=encoder)
    ckpt_path = ckpt.save(os.path.join(self._logging_dir, "ckpt"))
    return ckpt_path

  def test_task(self):
    ckpt_path = self._create_bert_ckpt()
    config = tfrbert_task.TFRBertConfig(
        output_preds=True,
        init_checkpoint=ckpt_path,
        aggregated_metrics=True,
        model=tfrbert_task.ModelConfig(
            encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(num_layers=1))))

    task = tfrbert_task.TFRBertTask(config, logging_dir=self._logging_dir)
    model = task.build_model()
    metrics = task.build_metrics()

    def _get_dataset(extra_features=None):

      def _dummy_data(_):
        dummy_word_ids = tf.random.uniform(
            minval=0, maxval=100, shape=(12, 10, 128), dtype=tf.int64)
        dummy_mask = tf.ones((10, 1, 128))
        dummy_type_ids = tf.zeros((10, 1, 128))
        dummy_example_list_mask = tf.ones((10, 1), dtype=tf.bool)
        x = dict(
            input_word_ids=dummy_word_ids,
            input_mask=dummy_mask,
            input_type_ids=dummy_type_ids,
            example_list_mask=dummy_example_list_mask)
        if extra_features:
          x.update({
              "query_id": tf.constant(
                  [[[1]], [[2]], [[1]], [[1]], [[2]],
                   [[3]], [[3]], [[2]], [[1]], [[3]]])
          })
          x.update({
              "document_id": tf.constant(
                  [[[101]], [[201]], [[102]], [[103]], [[202]],
                   [[301]], [[302]], [[203]], [[104]], [[303]]])
          })

        y = tf.constant(
            [[1], [1], [0], [0], [0], [1], [0], [0], [0], [0]], dtype=tf.int32)
        return x, y

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat(1)
      dataset = dataset.map(_dummy_data)
      return dataset

    task.build_inputs = _get_dataset
    train_dataset = task.build_inputs()
    vali_dataset = task.build_inputs(extra_features=["query_id", "document_id"])

    task.initialize(model)
    train_iterator = iter(train_dataset)
    vali_iterator = iter(vali_dataset)
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    task.train_step(next(train_iterator), model, optimizer, metrics=metrics)
    logs = task.validation_step(next(vali_iterator), model, metrics=metrics)
    logs = {x: (logs[x],) for x in logs}
    logs = task.aggregate_logs(step_outputs=logs)
    self.assertEqual(tf.constant(logs["query_id"]).shape, (1, 10, 1))
    self.assertEqual(tf.constant(logs["document_id"]).shape, (1, 10, 1))
    self.assertEqual(tf.constant(logs[tfrbert_task._PREDICTION]).shape,
                     (1, 10, 1))
    self.assertEqual(tf.constant(logs[tfrbert_task._LABEL]).shape, (1, 10, 1))
    metrics = task.reduce_aggregated_logs(logs)


if __name__ == "__main__":
  tf.test.main()
