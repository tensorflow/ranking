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

r"""The DASALC model in the following ICLR paper.

Zhen Qin, Le Yan, Honglei Zhuang, Yi Tay, Rama Kumar Pasumarthi, Xuanhui Wang,
Mike Bendersky, Marc Najork
"Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?"
ICLR 2021

The default hyperparameters set in this file was used on the Web30K dataset.
For Yahoo and Istella dataset we used the following configurations:

Yahoo:
train_batch_size: 256
learning_rate:0.0001
dropout_rate:0.7
input_noise_stddev:4.5
num_attention_heads: 4

Istella:
hidden_layer_dims:3072
input_noise_stddev:0.2
num_attention_layers: 2,

Note that the metrics reported in TF-Ranking are usually lower than the reported
numbers in the paper, since in TF-Ranking evaluation, queries with no relevant
docs will have zero ranking metrics, while such queries are ignored in the paper
evaluation, which is the norm in the literature.

The numbers reported in the paper were based models trained in a distributed
training environment. Due to the hardware difference and randomness, you may
still need to re-tune some of the hyperparameters.


The supported proto formats are listed at ../python/data.py.
--------------------------------------------------------------------------------
Sample command lines:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_numerical_elwc.tfrecord && \
EVAL=tensorflow_ranking/examples/data/vali_numerical_elwc.tfrecord && \
rm -rf $MODEL_DIR && \
bazel build -c opt \
tensorflow_ranking/research/dasalc_py_binary && \
./bazel-bin/tensorflow_ranking/research/dasalc_py_binary \
--train_input_pattern=$TRAIN \
--eval_input_pattern=$EVAL \
--model_dir=$MODEL_DIR

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
"""

from absl import flags

import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers
import tensorflow_ranking as tfr

flags.DEFINE_string("train_input_pattern", None,
                    "Input file path used for training.")
flags.DEFINE_string("eval_input_pattern", None,
                    "Input file path used for eval.")
flags.DEFINE_string("model_dir", None, "Output directory for models.")
flags.DEFINE_integer("batch_size", 128, "The batch size for train.")
flags.DEFINE_integer("num_train_steps", 15000, "Number of steps for train.")
flags.DEFINE_integer("num_eval_steps", 100, "Number of steps for evaluation.")
flags.DEFINE_integer("checkpoint_secs", 30,
                     "Saves a model checkpoint every checkpoint_secs seconds.")
flags.DEFINE_integer("num_checkpoints", 100,
                     "Saves at most num_checkpoints checkpoints in workspace.")
flags.DEFINE_integer("num_features", 136, "Number of features per example.")
flags.DEFINE_integer(
    "list_size", 200,
    "List size used for training. Use None for dynamic list size.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.4, "The dropout rate before output layer.")
flags.DEFINE_integer("hidden_layer_dims", 512,
                     "Number of units in each hidden layer.")
flags.DEFINE_string("loss", "softmax_loss",
                    "The RankingLossKey for the loss function.")
flags.DEFINE_float("batch_norm_moment", 0.9,
                   "Batch Normalization's momentum hyperparameter.")
flags.DEFINE_float("input_noise_stddev", 1.5,
                   "Input Gaussian noise standard deviation.")
flags.DEFINE_integer("num_attention_layers", 4,
                     "number of self attention layers.")
flags.DEFINE_integer("num_attention_heads", 2,
                     "number of self attention heads.")
flags.DEFINE_integer("head_size", 100, "Size of attention head.")

FLAGS = flags.FLAGS

_LABEL_FEATURE = "utility"
_CUTOFF = 30.
_MASK = "example_list_mask"
_PADDING_LABEL = -1


def context_feature_columns():
  """Returns context feature columns."""
  return {}


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = [
      "custom_features_{}".format(i + 1) for i in range(FLAGS.num_features)
  ]

  def log1p_cutoff(t):
    return tf.clip_by_value(
        tf.math.log1p(tf.abs(t)) * tf.sign(t), -_CUTOFF, _CUTOFF)

  example_feature_columns_ = {}
  for name in feature_names:
    example_feature_columns_[name] = tf.feature_column.numeric_column(
        name, shape=(1,), default_value=0.0, normalizer_fn=log1p_cutoff)

  return example_feature_columns_


def transform_function(features, mode):
  """Transform function for DASALC model."""
  mask = features.pop(_MASK)
  context_features, example_features = tfr.feature.encode_listwise_features(
      features=features,
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      mode=mode,
      scope="transform_layer")

  training = (mode == tf.estimator.ModeKeys.TRAIN)
  concat_tensor = tfr.keras.layers.ConcatFeatures()(
      inputs=(context_features, example_features, mask))
  din_layer = tfr.keras.layers.DocumentInteractionAttention(
      num_heads=FLAGS.num_attention_heads,
      head_size=FLAGS.head_size,
      num_layers=FLAGS.num_attention_layers,
      dropout=FLAGS.dropout_rate,
      input_noise_stddev=FLAGS.input_noise_stddev)
  example_features["document_interaction_network_embedding"] = din_layer(
      inputs=(concat_tensor, mask), training=training)
  return context_features, example_features


def scoring_function(context_features, example_features, mode):
  """A feed-forward network to score query-document pairs."""
  del context_features
  with tf.compat.v1.name_scope("input_layer"):
    input_features = [
        tf.compat.v1.layers.flatten(example_features[name])
        for name in sorted(example_features)
        if name != "document_interaction_network_embedding"
    ]
    input_layer = tf.concat(input_features, 1)
    context_input = tf.compat.v1.layers.flatten(
        example_features["document_interaction_network_embedding"])

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  cur_layer = tf.compat.v1.layers.batch_normalization(
      input_layer, training=is_training, momentum=FLAGS.batch_norm_moment)
  cur_layer = tf.keras.layers.GaussianNoise(FLAGS.input_noise_stddev)(
      cur_layer, training=is_training)

  context_layer = tf.compat.v1.layers.batch_normalization(
      context_input, training=is_training, momentum=FLAGS.batch_norm_moment)
  last_dim = FLAGS.hidden_layer_dims
  for layer_width in [FLAGS.hidden_layer_dims, FLAGS.hidden_layer_dims]:
    cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
    cur_layer = tf.nn.relu(cur_layer)
    cur_layer = tf.compat.v1.layers.batch_normalization(
        cur_layer, training=is_training, momentum=FLAGS.batch_norm_moment)
    cur_layer = tf.compat.v1.layers.dropout(
        inputs=cur_layer, rate=FLAGS.dropout_rate, training=is_training)

  cur_layer = tf.compat.v1.layers.dense(cur_layer, units=last_dim)
  context_layer = tf.compat.v1.layers.dense(context_layer, units=last_dim)
  output_layer = tf.math.multiply(cur_layer, context_layer)
  output_layer = tf.math.add(output_layer, cur_layer)
  output_layer = tf.nn.relu(output_layer)
  output_layer = tf.compat.v1.layers.dropout(
      inputs=output_layer, rate=FLAGS.dropout_rate, training=is_training)

  return tf.compat.v1.layers.dense(output_layer, units=1)


class DASALCPipeline(tfr.ext.pipeline.RankingPipeline):
  """A custom ranking pipeline for dasalc model."""

  def _make_serving_input_fn(self):
    """Returns `Estimator` `input_fn` for serving the model."""
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        example_feature_columns().values())

    serving_input_receiver_fn = (
        tfr.data.build_ranking_serving_input_receiver_fn(
            data_format="example_list_with_context",
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
            mask_feature_name=_MASK))

    return serving_input_receiver_fn

  def _make_dataset(self,
                    batch_size,
                    list_size,
                    input_pattern,
                    randomize_input=True,
                    num_epochs=None):
    """Overwrites the inner immplementation of input function.

    Args:
      batch_size: (int) The number of input examples to process per batch. Use
        params['batch_size'] for TPUEstimator, and `batch_size` for Estimator.
      list_size: (int) The list size for an ELWC example.
      input_pattern: (str) File pattern for the input data.
      randomize_input: (bool) If true, randomize input example order. It should
        almost always be true except for unittest/debug purposes.
      num_epochs: (int) The number of times the input dataset must be repeated.
        None to repeat the data indefinitely.

    Returns:
      A tuple of (feature tensors, label tensor).
    """
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        self._context_feature_columns.values())

    label_column = tf.feature_column.numeric_column(
        self._label_feature_name,
        dtype=self._label_feature_type,
        default_value=_PADDING_LABEL)
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(self._example_feature_columns.values()) + [label_column])

    dataset = tfr.data.build_ranking_dataset(
        file_pattern=input_pattern,
        data_format="example_list_with_context",
        batch_size=batch_size,
        list_size=list_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=randomize_input,
        num_epochs=num_epochs,
        prefetch_buffer_size=10000,
        reader_num_threads=64,
        mask_feature_name=_MASK)

    return dataset.map(self._features_and_labels)


def train_and_eval():
  """Train and Evaluate."""

  hparams = {
      "train_input_pattern": FLAGS.train_input_pattern,
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
      "listwise_inference": True,
      "convert_labels_to_binary": False,
      "model_dir": FLAGS.model_dir
  }

  optimizer = optimizers.get_optimizer_instance(
      "Adam", learning_rate=FLAGS.learning_rate)

  estimator = tfr.estimator.EstimatorBuilder(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      scoring_function=scoring_function,
      transform_function=transform_function,
      optimizer=optimizer,
      loss_reduction=tf.compat.v1.losses.Reduction.MEAN,
      hparams=hparams).make_estimator()

  ranking_pipeline = DASALCPipeline(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      hparams=hparams,
      estimator=estimator,
      label_feature_name="utility",
      label_feature_type=tf.int64,
      best_exporter_metric="metric/ndcg_5")

  ranking_pipeline.train_and_eval()


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_input_pattern")
  flags.mark_flag_as_required("eval_input_pattern")
  flags.mark_flag_as_required("model_dir")

  tf.compat.v1.app.run()
