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

r"""Keras Model to Estimator example code for proto formats stored in TFRecord.

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
tensorflow_ranking/examples/keras/keras_m2e_tfrecord_py_binary && \
./bazel-bin/tensorflow_ranking/examples/keras/keras_m2e_tfrecord_py_binary \
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
flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("eval_path", None, "Input file path used for eval.")
flags.DEFINE_string("vocab_path", None,
                    "Vocabulary path for query and document tokens.")
flags.DEFINE_string("model_dir", None, "Output directory for models.")
flags.DEFINE_integer("batch_size", 32, "The batch size for train.")
flags.DEFINE_integer("num_train_steps", 15000, "Number of steps for train.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.8, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["64", "32", "16"],
                  "Sizes for hidden layers.")
flags.DEFINE_integer(
    "list_size", None,
    "List size used for training. Use None for dynamic list size.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")
flags.DEFINE_string("loss", "approx_ndcg_loss",
                    "The RankingLossKey for the loss function.")
flags.DEFINE_string(
    "weights_feature_name", None,
    "The name of the feature where unbiased learning-to-rank "
    "weights are stored.")

FLAGS = flags.FLAGS

_LABEL_FEATURE = "relevance"
_PADDING_LABEL = -1
_EMBEDDING_DIMENSION = 20
_SIZE = "example_list_size"


def _get_feature_columns():
  """Returns context and example feature columns.

  Returns:
    A tuple of dicts (context_feature_columns, example_feature_columns), where
    the dicts are a mapping from feature name to feature column.
  """
  if FLAGS.vocab_path:
    sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="query_tokens", vocabulary_file=FLAGS.vocab_path)
  else:
    sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        key="query_tokens", hash_bucket_size=100)
  query_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  context_feature_columns = {"query_tokens": query_embedding_column}

  if FLAGS.vocab_path:
    sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="document_tokens", vocabulary_file=FLAGS.vocab_path)
  else:
    sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        key="document_tokens", hash_bucket_size=100)
  document_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  example_feature_columns = {"document_tokens": document_embedding_column}
  return context_feature_columns, example_feature_columns


def _get_example_weight_feature_column():
  if FLAGS.weights_feature_name:
    return tf.feature_column.numeric_column(
        FLAGS.weights_feature_name, dtype=tf.float32, default_value=1.)
  return None


def make_input_fn(file_pattern,
                  batch_size,
                  randomize_input=True,
                  num_epochs=None):
  """Returns `Estimator` `input_fn` for TRAIN and EVAL.

  Args:
    file_pattern: (string) file pattern for the TFRecord input data.
    batch_size: (int) number of input examples to process per batch.
    randomize_input: (bool) if true, randomize input example order. It should
      almost always be true except for unittest/debug purposes.
    num_epochs: (int) Number of times the input dataset must be repeated. None
      to repeat the data indefinitely.

  Returns:
    An `input_fn` for `Estimator`.
  """
  tf.compat.v1.logging.info("FLAGS.data_format={}".format(FLAGS.data_format))

  def _input_fn():
    """Defines the input_fn."""
    context_feature_columns, example_feature_columns = _get_feature_columns()
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        list(context_feature_columns.values()))

    label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
    weight_column = _get_example_weight_feature_column()
    example_fc_list = (
        list(example_feature_columns.values()) + [label_column] +
        ([weight_column] if weight_column else []))
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        example_fc_list)
    dataset = tfr.data.build_ranking_dataset(
        file_pattern=file_pattern,
        data_format=FLAGS.data_format,
        batch_size=batch_size,
        list_size=FLAGS.list_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=randomize_input,
        num_epochs=num_epochs,
        size_feature_name=_SIZE)
    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)

    return features, label

  return _input_fn


def make_serving_input_fn():
  """Returns serving input fn."""
  context_feature_columns, example_feature_columns = _get_feature_columns()
  context_feature_spec = tf.feature_column.make_parse_example_spec(
      context_feature_columns.values())
  example_feature_spec = tf.feature_column.make_parse_example_spec(
      example_feature_columns.values())
  return tfr.data.build_ranking_serving_input_receiver_fn(
      data_format=FLAGS.data_format,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      size_feature_name=_SIZE)


def get_estimator():
  """Create Keras ranking estimator."""
  context_feature_columns, example_feature_columns = _get_feature_columns()
  # To build your own custom ranking network, look at how canned
  # DNNRankingNetwork is implemented. You can subclass
  # tfr.keras.network.UnivariateRankingNetwork, or the more generic
  # tfr.keras.network.RankingNetwork to build your own network.
  network = tfr.keras.canned.DNNRankingNetwork(
      context_feature_columns=context_feature_columns,
      example_feature_columns=example_feature_columns,
      hidden_layer_dims=[int(d) for d in FLAGS.hidden_layer_dims],
      activation=tf.nn.relu,
      dropout=FLAGS.dropout_rate,
      use_batch_norm=True,
      batch_norm_moment=0.99,
      name="dnn_ranking_model")
  loss = tfr.keras.losses.get(
      FLAGS.loss, reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)
  metrics = tfr.keras.metrics.default_keras_metrics()
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=FLAGS.learning_rate)
  config = tf.estimator.RunConfig(save_checkpoints_steps=1000)
  ranker = tfr.keras.model.create_keras_model(
      network=network,
      loss=loss,
      metrics=metrics,
      optimizer=optimizer,
      size_feature_name=_SIZE)
  estimator = tfr.keras.estimator.model_to_estimator(
      model=ranker,
      model_dir=FLAGS.model_dir,
      config=config,
      weights_feature_name=FLAGS.weights_feature_name)

  return estimator


def train_and_eval():
  """Train and Evaluate."""
  train_input_fn = make_input_fn(FLAGS.train_path, FLAGS.batch_size)
  eval_input_fn = make_input_fn(
      FLAGS.eval_path, FLAGS.batch_size, randomize_input=False, num_epochs=1)

  estimator = get_estimator()
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
  exporters = tf.estimator.LatestExporter(
      "saved_model_exporter", serving_input_receiver_fn=make_serving_input_fn())
  eval_spec = tf.estimator.EvalSpec(
      name="eval",
      input_fn=eval_input_fn,
      steps=1,
      exporters=exporters,
      start_delay_secs=0,
      throttle_secs=15)

  # Train and validate.
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(_):
  tf.compat.v1.set_random_seed(1234)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_path")
  flags.mark_flag_as_required("eval_path")
  flags.mark_flag_as_required("model_dir")

  tf.compat.v1.app.run()
