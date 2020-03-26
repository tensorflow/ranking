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

r"""TF-Ranking example code for proto formats stored in TFRecord.

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
tensorflow_ranking/examples/tf_ranking_tfrecord_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_tfrecord_py_binary \
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
flags.DEFINE_string("weights_feature_name", "",
                    "The name of the feature where unbiased learning-to-rank "
                    "weights are stored.")
flags.DEFINE_bool("listwise_inference", False,
                  "If true, exports accept `data_format` while serving.")
flags.DEFINE_bool(
    "use_document_interactions", False,
    "If true, uses cross-document interactions to generate scores.")

FLAGS = flags.FLAGS

_LABEL_FEATURE = "relevance"
_PADDING_LABEL = -1
_EMBEDDING_DIMENSION = 20
_DOCUMENT_MASK = "__document_mask__"


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
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
    label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])
    dataset = tfr.data.build_ranking_dataset(
        file_pattern=file_pattern,
        data_format=FLAGS.data_format,
        batch_size=batch_size,
        list_size=FLAGS.list_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=randomize_input,
        num_epochs=num_epochs)
    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)

    # Add document_mask to features, which is True for valid documents and False
    # for invalid documents.
    if FLAGS.use_document_interactions:
      features[_DOCUMENT_MASK] = tf.not_equal(label, _PADDING_LABEL)

    return features, label

  return _input_fn


def make_serving_input_fn():
  """Returns serving input fn."""
  context_feature_spec = tf.feature_column.make_parse_example_spec(
      context_feature_columns().values())
  example_feature_spec = tf.feature_column.make_parse_example_spec(
      example_feature_columns().values())
  if FLAGS.listwise_inference:
    # Exports accept the specified FLAGS.data_format during serving.
    return tfr.data.build_ranking_serving_input_receiver_fn(
        data_format=FLAGS.data_format,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec)
  elif FLAGS.group_size == 1:
    # Exports accept tf.Example when group_size = 1.
    feature_spec = {}
    feature_spec.update(example_feature_spec)
    feature_spec.update(context_feature_spec)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)
  else:
    raise ValueError("FLAGS.group_size should be 1, but is {} when "
                     "FLAGS.export_listwise_inference is False".format(
                         FLAGS.group_size))


def _document_interaction_layer(context_features,
                                example_features,
                                document_mask=None):
  """A simplified demo version of Document Interaction Networks.

  Document Interaction Networks can capture cross-document interactions using
  self-attention mechanism. This simplified implementation (for demo purposes)
  uses a single-layer of self-attention, with a single head.

  For more details, please refer to the paper, "Self-Attentive Document
  Interaction Networks for Permutation Equivariant Ranking".
  https://arxiv.org/pdf/1910.09676.pdf

  Args:
    context_features: (dict) A mapping from feature name to 2D tensors of shape
      [batch_size, feature_dims].
    example_features: (dict) A mapping from feature name to 3D tensors of shape
      [batch_size, input_length, feature_dims].
    document_mask: A tensor of shape [batch_size, input_size], indicating the
      number of valid documents per query.

  Returns:
     document_interaction_embedding: A 3D tensor of size [batch_size,
     input_length, hidden_size].
  """
  sorted_example_keys = sorted(example_features.keys())
  example_inputs = tf.concat(
      [example_features[name] for name in sorted_example_keys], axis=2)

  outputs = [example_inputs]
  if context_features:
    sorted_context_keys = sorted(context_features.keys())
    context_inputs = tf.concat(
        [context_features[name] for name in sorted_context_keys], axis=1)
    context_inputs = tf.expand_dims(input=context_inputs, axis=1)
    length = tf.shape(input=example_inputs)[1]
    context_inputs = tf.tile(input=context_inputs, multiples=[1, length, 1])
    outputs.append(context_inputs)

  document_embedding = tf.concat(outputs, axis=2)
  document_interaction_embedding = tf.keras.layers.Attention()(
      [document_embedding, document_embedding],
      [document_mask, document_mask])

  return document_interaction_embedding


def make_transform_fn():
  """Returns a transform_fn that converts features to dense Tensors."""

  def _transform_fn(features, mode):
    """Defines transform_fn."""
    if mode == tf.estimator.ModeKeys.PREDICT and not FLAGS.listwise_inference:
      # We expect tf.Example as input during serving. In this case, group_size
      # must be set to 1.
      if FLAGS.group_size != 1:
        raise ValueError(
            "group_size should be 1 to be able to export model, but get %s" %
            FLAGS.group_size)
      context_features, example_features = (
          tfr.feature.encode_pointwise_features(
              features=features,
              context_feature_columns=context_feature_columns(),
              example_feature_columns=example_feature_columns(),
              mode=mode,
              scope="transform_layer"))
    else:
      context_features, example_features = tfr.feature.encode_listwise_features(
          features=features,
          context_feature_columns=context_feature_columns(),
          example_feature_columns=example_feature_columns(),
          mode=mode,
          scope="transform_layer")

    # Document Interactions Layer.
    if FLAGS.use_document_interactions:
      if mode == tf.estimator.ModeKeys.PREDICT:
        if not FLAGS.listwise_inference:
          raise ValueError("Only listwise inference is compatible for networks "
                           "with document interactions.")
        else:
          # Set document mask to None. This assumes document list for each
          # query has no padding. It is recommended to pass batches of queries
          # with the same number of documents during inference.
          features[_DOCUMENT_MASK] = None
      document_mask = features.pop(_DOCUMENT_MASK)
      document_interaction_embedding = _document_interaction_layer(
          context_features, example_features, document_mask=document_mask)
      example_features[
          "document_interaction_embedding"] = document_interaction_embedding
    return context_features, example_features

  return _transform_fn


def make_score_fn():
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a group of documents."""
    del [params, config]
    with tf.compat.v1.name_scope("input_layer"):
      context_input = [
          tf.compat.v1.layers.flatten(context_features[name])
          for name in sorted(context_feature_columns())
      ]
      group_input = [
          tf.compat.v1.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns(use_weight_feature=False))
      ]
      if FLAGS.use_document_interactions:
        group_input.append(
            tf.compat.v1.layers.flatten(
                group_features["document_interaction_embedding"]))
      input_layer = tf.concat(context_input + group_input, 1)
      tf.compat.v1.summary.scalar("input_sparsity",
                                  tf.nn.zero_fraction(input_layer))
      tf.compat.v1.summary.scalar("input_max",
                                  tf.reduce_max(input_tensor=input_layer))
      tf.compat.v1.summary.scalar("input_min",
                                  tf.reduce_min(input_tensor=input_layer))
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = input_layer
    cur_layer = tf.compat.v1.layers.batch_normalization(
        cur_layer, training=is_training, momentum=0.99)

    for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
          cur_layer, training=is_training, momentum=0.99)
      cur_layer = tf.nn.relu(cur_layer)
      tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                  tf.nn.zero_fraction(cur_layer))
      cur_layer = tf.compat.v1.layers.dropout(
          inputs=cur_layer, rate=FLAGS.dropout_rate, training=is_training)
    logits = tf.compat.v1.layers.dense(cur_layer, units=FLAGS.group_size)
    return logits

  return _score_fn


def eval_metric_fns():
  """Returns a dict from name to metric functions."""
  metric_fns = {}
  metric_fns.update({
      "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
          tfr.metrics.RankingMetricKey.ARP,
          tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
      ]
  })
  metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10]
  })
  for topn in [1, 3, 5, 10]:
    metric_fns["metric/weighted_ndcg@%d" % topn] = (
        tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG,
            weights_feature_name=FLAGS.weights_feature_name, topn=topn))
  return metric_fns


def train_and_eval():
  """Train and Evaluate."""
  train_input_fn = make_input_fn(FLAGS.train_path, FLAGS.batch_size)
  eval_input_fn = make_input_fn(
      FLAGS.eval_path, FLAGS.batch_size, randomize_input=False, num_epochs=1)

  optimizer = tf.compat.v1.train.AdagradOptimizer(
      learning_rate=FLAGS.learning_rate)

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    minimize_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())
    train_op = tf.group([minimize_op, update_ops])
    return train_op

  ranking_head = tfr.head.create_ranking_head(
      loss_fn=tfr.losses.make_loss_fn(
          FLAGS.loss,
          weights_feature_name=FLAGS.weights_feature_name),
      eval_metric_fns=eval_metric_fns(),
      train_op_fn=_train_op_fn)

  estimator = tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=FLAGS.group_size,
          transform_fn=make_transform_fn(),
          ranking_head=ranking_head),
      model_dir=FLAGS.model_dir,
      config=tf.estimator.RunConfig(save_checkpoints_steps=1000))

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
