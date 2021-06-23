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

r"""TFR-BERT model."""

from typing import Dict

import tensorflow as tf

from tensorflow_ranking.python.keras import model as tfr_model

TensorLike = tf.types.experimental.TensorLike
TensorDict = Dict[str, TensorLike]


# TODO: Extend to more generic encoders in addition to BERT.
class TFRBertScorer(tfr_model.UnivariateScorer):
  """Univariate BERT-based scorer."""

  def __init__(self,
               encoder: tf.keras.Model,
               bert_output_dropout: float,
               name: str = "tfrbert",
               **kwargs):
    self.encoder = encoder
    self._dropout_layer = tf.keras.layers.Dropout(rate=bert_output_dropout)
    self._score_layer = tf.keras.layers.Dense(units=1, name="score")

  def _score_flattened(
      self,
      context_features: TensorDict,
      example_features: TensorDict,
  ) -> tf.Tensor:
    """See `UnivariateScorer`."""
    # See BertEncoder class for `encoder` outputs.
    bert_output = self.encoder(example_features)
    pooled_output = bert_output["pooled_output"]
    output = self._dropout_layer(pooled_output)
    return self._score_layer(output)


class TFRBertModelBuilder(tfr_model.ModelBuilder):
  """Model builder for TFR-BERT models."""

  def build(self) -> tf.keras.Model:
    model = super().build()
    model.checkpoint_items = {"encoder": self._scorer.encoder}
    return model
