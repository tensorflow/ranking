# Copyright 2020 The TensorFlow Ranking Authors.
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

r"""TF-Ranking example code for training a canned DNN estimator.

The supported proto formats are listed at ../python/data.py.
--------------------------------------------------------------------------------
Sample command lines:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_elwc.tfrecord && \
EVAL=tensorflow_ranking/examples/data/eval_elwc.tfrecord && \
VOCAB=tensorflow_ranking/examples/data/vocab.txt && \
WEIGHT_FEATURE_NAME="doc_weight" && \
rm -rf $MODEL_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_canned_dnn_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_canned_dnn_py_binary \
--train_path=$TRAIN \
--eval_path=$EVAL \
--vocab_path=$VOCAB \
--model_dir=$MODEL_DIR \
--data_format=example_list_with_context \
--weights_feature_name=$WEIGHT_FEATURE_NAME

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
"""

from absl import flags

import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_enum(
    "data_format", "example_list_with_context",
    ["example_list_with_context", "example_in_example", "sequence_example"],
    "Data format defined in data.py.")
flags.DEFINE_string("train_input_pattern", None,
                    "Input file path used for training.")
flags.DEFINE_string("eval_input_pattern", None,
                    "Input file path used for eval.")
flags.DEFINE_string("vocab_path", None,
                    "Vocabulary path for query and document tokens.")
flags.DEFINE_string("model_dir", None, "Output directory for models.")
flags.DEFINE_integer("batch_size", 32, "The batch size for train.")
flags.DEFINE_integer("num_train_steps", 15000, "Number of steps for train.")
flags.DEFINE_integer("num_eval_steps", 10, "Number of steps for evaluation.")
flags.DEFINE_integer("checkpoint_secs", 30,
                     "Saves a model checkpoint every checkpoint_secs seconds.")
flags.DEFINE_integer("num_checkpoints", 100,
                     "Saves at most num_checkpoints checkpoints in workspace.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate for optimizer.")
flags.DEFINE_float("dropout", 0.8, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["64", "32", "16"],
                  "Sizes for hidden layers.")
flags.DEFINE_integer(
    "list_size", None,
    "List size used for training. Use None for dynamic list size.")
flags.DEFINE_string("loss", "approx_ndcg_loss",
                    "The RankingLossKey for the loss function.")
flags.DEFINE_string(
    "weights_feature_name", "",
    "The name of the feature where unbiased learning-to-rank "
    "weights are stored.")
flags.DEFINE_bool("convert_labels_to_binary", False,
                  "If true, relevance labels are set to either 0 or 1.")
flags.DEFINE_bool("listwise_inference", False,
                  "If true, exports accept `data_format` while serving.")

FLAGS = flags.FLAGS

_LABEL_FEATURE = "relevance"
_EMBEDDING_DIMENSION = 20


def context_feature_columns():
  """Returns context feature names to column definitions."""
  if FLAGS.vocab_path:
    sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="query_tokens", vocabulary_file=FLAGS.vocab_path)
  else:
    sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        key="query_tokens", hash_bucket_size=100)
  query_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  return {"query_tokens": query_embedding_column}


def example_feature_columns(use_weight_feature=True):
  """Returns the example feature columns."""
  if FLAGS.vocab_path:
    sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="document_tokens", vocabulary_file=FLAGS.vocab_path)
  else:
    sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        key="document_tokens", hash_bucket_size=100)
  document_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  feature_columns = {"document_tokens": document_embedding_column}
  if use_weight_feature and FLAGS.weights_feature_name:
    feature_columns[FLAGS.weights_feature_name] = (
        tf.feature_column.numeric_column(FLAGS.weights_feature_name,
                                         default_value=1.))
  return feature_columns


def train_and_eval():
  """Train and Evaluate."""
  optimizer = tf.compat.v1.train.AdagradOptimizer(
      learning_rate=FLAGS.learning_rate)

  estimator = tfr.estimator.make_dnn_ranking_estimator(
      example_feature_columns(),
      FLAGS.hidden_layer_dims,
      context_feature_columns=context_feature_columns(),
      optimizer=optimizer,
      learning_rate=FLAGS.learning_rate,
      loss=FLAGS.loss,
      loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
      activation_fn=tf.nn.relu,
      dropout=FLAGS.dropout,
      use_batch_norm=True,
      model_dir=FLAGS.model_dir)

  hparams = {"train_input_pattern": FLAGS.train_input_pattern,
             "eval_input_pattern": FLAGS.eval_input_pattern,
             "learning_rate": FLAGS.learning_rate,
             "train_batch_size": FLAGS.batch_size,
             "eval_batch_size": FLAGS.batch_size,
             "predict_batch_size": FLAGS.batch_size,
             "num_train_steps": FLAGS.num_train_steps,
             "num_eval_steps": FLAGS.num_eval_steps,
             "checkpoint_secs": FLAGS.checkpoint_secs,
             "num_checkpoints": FLAGS.num_checkpoints,
             "loss": FLAGS.loss,
             "list_size": FLAGS.list_size,
             "convert_labels_to_binary": FLAGS.convert_labels_to_binary,
             "listwise_inference": FLAGS.listwise_inference,
             "model_dir": FLAGS.model_dir}

  ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
      context_feature_columns(),
      example_feature_columns(),
      hparams,
      estimator=estimator,
      label_feature_name=_LABEL_FEATURE,
      label_feature_type=tf.int64)

  ranking_pipeline.train_and_eval()


def main(_):
  tf.compat.v1.set_random_seed(1234)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_path")
  flags.mark_flag_as_required("eval_path")
  flags.mark_flag_as_required("model_dir")

  tf.compat.v1.app.run()
