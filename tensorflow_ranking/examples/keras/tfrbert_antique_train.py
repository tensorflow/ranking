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

r"""A training driver which fine-tunes a TFR-BERT model.

Please download a BERT checkpoint from tensorflow models
website: https://github.com/tensorflow/models/tree/master/official/nlp/bert.
Note that those checkpoints are TF 2.x compatible, which are different from the
checkpoints downloaded here: https://github.com/google-research/bert. You may
convert a TF 1.x checkpoint to TF 2.x using `tf2_encoder_checkpoint_converter`
under https://github.com/tensorflow/models/tree/master/official/nlp/bert.
The following command downloads an uncased BERT-base model checkpoint for you:

```
mkdir -p /tmp/tfrbert && \
wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/v3/\
uncased_L-12_H-768_A-12.tar.gz -P /tmp/tfrbert  && \
mkdir -p /tmp/tfrbert/uncased_L-12_H-768_A-12 && \
tar -xvf /tmp/tfrbert/uncased_L-12_H-768_A-12.tar.gz \
--strip-components 3 -C /tmp/tfrbert/uncased_L-12_H-768_A-12/
```

You can also download from here the Antique data set which has been converted
into BERT-compatible format:
https://ciir.cs.umass.edu/downloads/Antique/tfr-bert/ELWC/. The following
command downloads the data set to "/tmp/tfrbert/data/" directory.

```
mkdir -p /tmp/tfrbert/data && \
wget https://ciir.cs.umass.edu/downloads/Antique/tf-ranking/\
antique_train_seq_64_elwc.tfrecords -P /tmp/tfrbert/data && \
wget https://ciir.cs.umass.edu/downloads/Antique/tf-ranking/\
antique_test_seq_64_elwc.tfrecords -P /tmp/tfrbert/data
```

Then, use the following command to run training and evaluation locally with CPU
or GPU. For GPU, please add `CUDA_VISIBLE_DEVICES=0` and `--config=cuda`. The
example toy data contains 3 lists in train and test respectively. Due to the
large number of BERT parameters, if running into the `out-of-memory` issue,
plese see: https://github.com/google-research/bert#out-of-memory-issues.

MODEL_DIR="/tmp/tfrbert/model" && \
bazel build -c opt \
tensorflow_ranking/examples/keras:tfrbert_antique_train && \
./bazel-bin/tensorflow_ranking/examples/keras/tfrbert_antique_train \
  --experiment="tfr_bert" \
  --mode="train_and_eval" \
  --model_dir="${MODEL_DIR}" \
  --config_file=\
tensorflow_ranking/examples/keras/tfrbert_antique_train_config.yaml

Change the paramters in the .yaml `config_file` if you want to change the BERT
checkpoint, the data set and the training configurations. Change the `mode` to
"eval" and modify the `output_preds` in the .yaml config file to true to obtain
prediction of a trained model.
"""

from absl import app
from absl import flags
import gin

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
# pylint: disable=unused-import
from tensorflow_ranking.python.keras.tfrbert import experiments
# pylint: enable=unused-import

FLAGS = flags.FLAGS


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu,
      **params.runtime.model_parallelism())
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
