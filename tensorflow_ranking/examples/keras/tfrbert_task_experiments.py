# Copyright 2024 The TensorFlow Ranking Authors.
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

"""TFR-BERT experiment configurations."""
# pylint: disable=g-doc-return-or-yield,line-too-long

from tensorflow_ranking.extension import premade

# pylint: disable=g-import-not-at-top
try:
  from official.core import config_definitions as cfg
  from official.core import exp_factory
  from official.modeling import optimization
except ModuleNotFoundError:
  raise ModuleNotFoundError(
      'tf-models-official needs to be installed. Run command: '
      '`pip install tf-models-official`.') from None
# pylint: enable=g-import-not-at-top


@exp_factory.register_config_factory('tfr_bert')
def tfrbert_exp() -> cfg.ExperimentConfig:
  """Defines a TFR-BERT experiment."""
  config = cfg.ExperimentConfig(
      task=premade.TFRBertConfig(
          train_data=premade.TFRBertDataConfig(),
          validation_data=premade.TFRBertDataConfig(
              is_training=False, drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 3e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  config.task.model.encoder.type = 'bert'
  return config
