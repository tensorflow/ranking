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

r"""An example of running a TF ranking model with the RankingPipeline.

This ranking model is the same as the `tf_ranking_record` example but without
document interactions. Here, one only needs to define `context_feature_columns`,
`example_feature_columns` and `scoring_function`.

The supported proto format is ExampleListWithContext.
--------------------------------------------------------------------------------

A sample command line run is provided in below:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_elwc.tfrecord && \
EVAL=tensorflow_ranking/examples/data/eval_elwc.tfrecord && \
VOCAB=tensorflow_ranking/examples/data/vocab.txt && \
rm -rf $MODEL_DIR && \
bazel build -c opt \
tensorflow_ranking/extension/examples/pipeline_example_py_binary && \
./bazel-bin/tensorflow_ranking/extension/examples/pipeline_example_py_binary \
--train_input_pattern=$TRAIN \
--eval_input_pattern=$EVAL \
--vocab_path=$VOCAB \
--model_dir=$MODEL_DIR \
--num_eval_steps=10 \
--list_size=5

Refer to `tensorflow_ranking.extension.pipeline_flags` for more flag options.

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
"""

from absl import flags

import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_string("train_input_pattern", "",
                    "Input file path pattern used for training.")
flags.DEFINE_string("eval_input_pattern", "",
                    "Input file path pattern used for eval.")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate for the optimizer.")
flags.DEFINE_integer("train_batch_size", 8,
                     "Number of input records used per batch for training.")
flags.DEFINE_integer("eval_batch_size", 8,
                     "Number of input records used per batch for eval.")
flags.DEFINE_integer("checkpoint_secs", 120,
                     "Saves a model checkpoint every checkpoint_secs seconds.")
flags.DEFINE_integer("num_checkpoints", 1000,
                     "Saves at most num_checkpoints checkpoints in workspace.")
flags.DEFINE_integer(
    "num_train_steps", 200000,
    "Number of training iterations. Default means continuous training.")
flags.DEFINE_integer("num_eval_steps", 100, "Number of evaluation iterations.")
flags.DEFINE_string(
    "loss", "softmax_loss",
    "The RankingLossKey deciding the loss function used in training.")
flags.DEFINE_integer("list_size", None, "List size used for training.")
flags.DEFINE_bool(
    "listwise_inference", False,
    "Whether the inference will be performed with the listwise data format "
    "such as the `ExampleListWithContext`.")
flags.DEFINE_bool("convert_labels_to_binary", False,
                  "If true, relevance labels are set to either 0 or 1.")
flags.DEFINE_string("model_dir", None, "Output directory for models.")
flags.DEFINE_string("hidden_layer_dims", "64,32,8",
                    "Number of units in each hidden layer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate.")
flags.DEFINE_float("batch_normalization_momentum", 0.4,
                   "Batch Normalization's momentum hyperparameter.")
flags.DEFINE_integer("num_features", 136, "Number of features.")
flags.DEFINE_string("vocab_path", None,
                    "Vocabulary path for query and document tokens.")

FLAGS = flags.FLAGS

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


def example_feature_columns():
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
  return feature_columns


def scoring_function(context_features, example_features, mode):
  """A feed-forward network to score query-document pairs."""
  with tf.compat.v1.name_scope("input_layer"):
    context_input = [
        tf.compat.v1.layers.flatten(context_features[name])
        for name in sorted(context_feature_columns())
    ]
    example_input = [
        tf.compat.v1.layers.flatten(example_features[name])
        for name in sorted(example_feature_columns())
    ]
    input_layer = tf.concat(context_input + example_input, 1)

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  cur_layer = input_layer
  cur_layer = tf.compat.v1.layers.batch_normalization(
      cur_layer, training=is_training, momentum=0.99)

  layer_dims = [int(d) for d in FLAGS.hidden_layer_dims.split(",")]
  for layer_width in layer_dims:
    cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
    cur_layer = tf.compat.v1.layers.batch_normalization(
        cur_layer, training=is_training, momentum=0.99)
    cur_layer = tf.nn.relu(cur_layer)
    cur_layer = tf.compat.v1.layers.dropout(
        inputs=cur_layer, rate=FLAGS.dropout_rate, training=is_training)
  return tf.compat.v1.layers.dense(cur_layer, units=1)


def train_and_eval():
  """Runs training and evaluation with the `RankingPipeline`."""
  # The below contains a set of common flags for a TF-Ranking model. You need to
  # include all of them for adopting the `RankingPipeline`.
  hparams = dict(
      train_input_pattern=FLAGS.train_input_pattern,
      eval_input_pattern=FLAGS.eval_input_pattern,
      learning_rate=FLAGS.learning_rate,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      checkpoint_secs=FLAGS.checkpoint_secs,
      num_checkpoints=FLAGS.num_checkpoints,
      num_train_steps=FLAGS.num_train_steps,
      num_eval_steps=FLAGS.num_eval_steps,
      loss=FLAGS.loss,
      list_size=FLAGS.list_size,
      listwise_inference=FLAGS.listwise_inference,
      convert_labels_to_binary=FLAGS.convert_labels_to_binary,
      model_dir=FLAGS.model_dir)

  estimator = tfr.estimator.EstimatorBuilder(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      scoring_function=scoring_function,
      hparams=hparams).make_estimator()

  ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      hparams=hparams,
      estimator=estimator,
      label_feature_name="relevance",
      label_feature_type=tf.int64,
      best_exporter_metric="metric/ndcg_5")

  ranking_pipeline.train_and_eval()


def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.compat.v1.app.run()
