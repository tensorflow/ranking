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

"""Tests for tfrbert_model.py."""

import tensorflow as tf
from official.nlp.configs import encoders
from tensorflow_ranking.python.keras import model as tfr_model
from tensorflow_ranking.python.keras.tfrbert import tfrbert_model


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
    scorer = tfrbert_model.TFRBertScorer(
        encoder=encoder_network,
        bert_output_dropout=0.1)

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
        mask_feature_name='example_list_mask',
        name='tfrbert_model')
    model = model_builder.build()

    output = model(self._create_input_data())
    self.assertAllEqual(output.shape.as_list(), [12, 10])


if __name__ == '__main__':
  tf.test.main()
