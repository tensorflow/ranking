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

# Lint as: python3
r"""Keras Ranking Pipeline example for proto formats stored in TFRecord.

The supported proto formats are listed at ../python/data.py.
--------------------------------------------------------------------------------
Sample command lines:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_numerical_elwc.tfrecord && \
VALIDATION=tensorflow_ranking/examples/data/vali_numerical_elwc.tfrecord && \
TEST=tensorflow_ranking/examples/data/test_numerical_elwc.tfrecord && \
rm -rf "${MODEL_DIR}" && \
bazel build -c opt \
tensorflow_ranking/examples/keras/keras_dnn_tfrecord.par && \
./bazel-bin/tensorflow_ranking/examples/keras/keras_dnn_tfrecord.par \
   --train_input_pattern ${TRAIN} \
   --valid_input_pattern ${VALIDATION} \
   --test_input_pattern ${TEST} \
   --num_train_steps 1000 \
   --num_valid_steps 100 \
   --num_epochs 5 \
   --learning_rate 0.05 \
   --hidden_layer_dims "64,32,16" \
   --train_batch_size 32 \
   --dropout 0.4 \
   --model_dir ${OUTPUT_DIR} \

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
"""
from typing import Dict, Tuple

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_enum(
    "strategy", "MirroredStrategy",
    ["MirroredStrategy", "MultiWorkerMirroredStrategy"],
    "Choose tf.distribute strategy to use.")

flags.DEFINE_string(
    "model_dir", None, "The directory where the model weights and "
    "training/validation summaries are stored.")

flags.DEFINE_string(
    "train_input_pattern", None,
    "Input file path pattern used for training.")

flags.DEFINE_string(
    "valid_input_pattern", None,
    "Input file path pattern used for validation.")

flags.DEFINE_string(
    "test_input_pattern", None,
    "Input file path pattern used for test.")

flags.DEFINE_string("optimizer", "adagrad",
                    "Optimizer key, see `tf.keras.optimizers`.")

flags.DEFINE_float("learning_rate", 0.005, "Learning rate for optimizer.")

flags.DEFINE_bool("convert_labels_to_binary", False,
                  "Whether to use binary label.")

flags.DEFINE_float("dropout", 0.5, "The dropout rate before output layer.")

flags.DEFINE_bool("use_batch_norm", True, "Whether to use batch normalization.")

flags.DEFINE_float(
    "batch_norm_moment", 0.99,
    "Used when use_batch_norm=True. Batch Normalization moment parameter.")

flags.DEFINE_string("hidden_layer_dims", "128",
                    "Number of units in each hidden layer.")

flags.DEFINE_integer("train_batch_size", 16,
                     "Number of input records used per batch for training.")

flags.DEFINE_integer("valid_batch_size", 32,
                     "Number of input records used per batch for validation.")

flags.DEFINE_integer("num_train_steps", 100000,
                     "Number of training iterations.")

flags.DEFINE_integer("num_valid_steps", 100, "Number of validation iterations.")

flags.DEFINE_integer("num_epochs", 100,
                     "Number of passes over the training data.")

flags.DEFINE_integer(
    "steps_per_execution", 10, "The number of batches to run during each "
    "`tf.function` call. Running multiple batches inside a single `tf.function`"
    " call can greatly improve performance.")

flags.DEFINE_integer("list_size", None, "List size used for training.")

flags.DEFINE_integer("num_features", 136, "Number of features.")

flags.DEFINE_string("loss", "approx_ndcg_loss",
                    "See tfr.losses.RankingLossKey.")

flags.DEFINE_bool("use_log1p", False, "Whether to apply log1p feature "
                  "transformation.")

flags.DEFINE_bool("export_best_model", False,
                  "Whether to export the best model.")

FLAGS = flags.FLAGS

# The document relevance label.
_LABEL_FEATURE = "utility"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1.
_MASK = "example_list_mask"


def _create_feature_spec() -> Tuple[Dict[str, tf.io.FixedLenFeature], Dict[
    str, tf.io.FixedLenFeature], Tuple[str, tf.io.FixedLenFeature]]:
  """Create context and example feature spec for data parsing.

  Returns:
    (context feature specs, example feature specs, label spec).
  """
  context_feature_spec = {}
  example_feature_spec = {
      "custom_features_{}".format(i + 1):
      tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0.0)
      for i in range(0, FLAGS.num_features)
  }
  label_spec = (_LABEL_FEATURE,
                tf.io.FixedLenFeature(
                    shape=(1,), dtype=tf.float32, default_value=_PADDING_LABEL))
  return context_feature_spec, example_feature_spec, label_spec


def train_and_eval():
  """Trains and validates the Keras Model with the pipeline."""
  dataset_hparams = tfr.keras.pipeline.DatasetHparams(
      train_input_pattern=FLAGS.train_input_pattern,
      valid_input_pattern=FLAGS.valid_input_pattern,
      train_batch_size=FLAGS.train_batch_size,
      valid_batch_size=FLAGS.valid_batch_size,
      list_size=FLAGS.list_size,
      dataset_reader=tf.data.TFRecordDataset,
      convert_labels_to_binary=FLAGS.convert_labels_to_binary)
  pipeline_hparams = tfr.keras.pipeline.PipelineHparams(
      model_dir=FLAGS.model_dir,
      num_epochs=FLAGS.num_epochs,
      steps_per_epoch=(FLAGS.num_train_steps // FLAGS.num_epochs),
      validation_steps=FLAGS.num_valid_steps,
      loss=FLAGS.loss,
      loss_reduction=tf.losses.Reduction.AUTO,
      optimizer=FLAGS.optimizer,
      learning_rate=FLAGS.learning_rate,
      steps_per_execution=FLAGS.steps_per_execution,
      export_best_model=FLAGS.export_best_model,
      strategy=FLAGS.strategy)

  context_feature_spec, example_feature_spec, label_spec = _create_feature_spec(
  )

  preprocess_dict = {}
  if FLAGS.use_log1p:
    preprocess_dict = {
        fname: lambda t: tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
        for fname in example_feature_spec.keys()
    }

  dnn_scorer = tfr.keras.model.DNNScorer(
      hidden_layer_dims=map(int, FLAGS.hidden_layer_dims.split(",")),
      output_units=1,
      activation=tf.nn.relu,
      input_batch_norm=FLAGS.use_batch_norm,
      use_batch_norm=FLAGS.use_batch_norm,
      batch_norm_moment=FLAGS.batch_norm_moment,
      dropout=FLAGS.dropout)

  model_builder = tfr.keras.model.ModelBuilder(
      input_creator=tfr.keras.model.FeatureSpecInputCreator(
          context_feature_spec, example_feature_spec),
      preprocessor=tfr.keras.model.PreprocessorWithSpec(preprocess_dict),
      scorer=dnn_scorer,
      mask_feature_name=_MASK,
      name="keras_dnn_model")

  ranking_pipeline = tfr.keras.pipeline.SimplePipeline(
      model_builder=model_builder,
      dataset_builder=tfr.keras.pipeline.SimpleDatasetBuilder(
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          mask_feature_name=_MASK,
          label_spec=label_spec,
          hparams=dataset_hparams),
      hparams=pipeline_hparams)

  ranking_pipeline.train_and_validate()


def main(_):
  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_input_pattern")
  flags.mark_flag_as_required("valid_input_pattern")
  app.run(main)
