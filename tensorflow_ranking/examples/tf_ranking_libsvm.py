# Copyright 2018 The TensorFlow Ranking Authors.
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

r"""TF Ranking sample code for LETOR datasets in LibSVM format.

WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.

A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:

<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76

In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------

Sample command lines:

OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136

You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
"""

from absl import flags

import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 100000, "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["256", "128", "64"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string("loss", "pairwise_logistic_loss",
                    "The RankingLossKey for loss function.")

FLAGS = flags.FLAGS


class IteratorInitializerHook(tf.train.SessionRunHook):
  """Hook to initialize data iterator after session is created."""

  def __init__(self):
    super(IteratorInitializerHook, self).__init__()
    self.iterator_initializer_fn = None

  def after_create_session(self, session, coord):
    """Initialize the iterator after the session has been created."""
    del coord
    self.iterator_initializer_fn(session)


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = ["{}".format(i + 1) for i in range(FLAGS.num_features)]
  return {
      name: tf.feature_column.numeric_column(
          name, shape=(1,), default_value=0.0) for name in feature_names
  }


def load_libsvm_data(path, list_size):
  """Returns features and labels in numpy.array."""

  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  tf.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of quries.
  feature_map = {k: [] for k in example_feature_columns()}
  label_list = []
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:
    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0
        for k in feature_map:
          feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
        label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1
      # Keep the first 'list_size' docs only.
      if doc_idx >= list_size:
        discarded_docs += 1
        continue
      for k, v in six.iteritems(features):
        assert k in feature_map, "Key {} not founded in features.".format(k)
        feature_map[k][batch_idx][doc_idx, 0] = v
      label_list[batch_idx][doc_idx] = label

  tf.logging.info("Number of queries: {}".format(len(qid_to_index)))
  tf.logging.info("Number of documents in total: {}".format(total_docs))
  tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

  # Convert everything to np.array.
  for k in feature_map:
    feature_map[k] = np.array(feature_map[k])
  return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
  """Set up training input in batches."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _train_input_fn():
    """Defines training input fn."""
    features_placeholder = {
        k: tf.placeholder(v.dtype, v.shape) for k, v in six.iteritems(features)
    }
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,
                                                  labels_placeholder))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()

  return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(features, labels):
  """Set up eval inputs in a single batch."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _eval_input_fn():
    """Defines eval input fn."""
    features_placeholder = {
        k: tf.placeholder(v.dtype, v.shape) for k, v in six.iteritems(features)
    }
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensors((features_placeholder,
                                            labels_placeholder))
    iterator = dataset.make_initializable_iterator()
    feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()

  return _eval_input_fn, iterator_initializer_hook


def make_score_fn():
  """Returns a groupwise score fn to build `EstimatorSpec`."""

  def _score_fn(unused_context_features, group_features, mode, unused_params,
                unused_config):
    """Defines the network to score a group of documents."""
    with tf.name_scope("input_layer"):
      group_input = [
          tf.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns())
      ]
      input_layer = tf.concat(group_input, 1)
      tf.summary.scalar("input_sparsity", tf.nn.zero_fraction(input_layer))
      tf.summary.scalar("input_max", tf.reduce_max(input_layer))
      tf.summary.scalar("input_min", tf.reduce_min(input_layer))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = tf.layers.batch_normalization(input_layer, training=is_training)
    for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
      cur_layer = tf.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
      cur_layer = tf.nn.relu(cur_layer)
      tf.summary.scalar("fully_connected_{}_sparsity".format(i),
                        tf.nn.zero_fraction(cur_layer))
    cur_layer = tf.layers.dropout(
        cur_layer, rate=FLAGS.dropout_rate, training=is_training)
    logits = tf.layers.dense(cur_layer, units=FLAGS.group_size)
    return logits

  return _score_fn


def get_eval_metric_fns():
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
  return metric_fns


def train_and_eval():
  """Train and Evaluate."""

  features, labels = load_libsvm_data(FLAGS.train_path, FLAGS.list_size)
  train_input_fn, train_hook = get_train_inputs(features, labels,
                                                FLAGS.train_batch_size)

  features_vali, labels_vali = load_libsvm_data(FLAGS.vali_path,
                                                FLAGS.list_size)
  vali_input_fn, vali_hook = get_eval_inputs(features_vali, labels_vali)

  features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                FLAGS.list_size)
  test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=FLAGS.learning_rate,
        optimizer="Adagrad")

  ranking_head = tfr.head.create_ranking_head(
      loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
      eval_metric_fns=get_eval_metric_fns(),
      train_op_fn=_train_op_fn)

  estimator = tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=FLAGS.group_size,
          transform_fn=None,
          ranking_head=ranking_head),
      config=tf.estimator.RunConfig(
          FLAGS.output_dir, save_checkpoints_steps=1000))

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      hooks=[train_hook],
      max_steps=FLAGS.num_train_steps)
  vali_spec = tf.estimator.EvalSpec(
      input_fn=vali_input_fn,
      hooks=[vali_hook],
      steps=1,
      start_delay_secs=0,
      throttle_secs=30)

  # Train and validate
  tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

  # Evaluate on the test data.
  estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_path")
  flags.mark_flag_as_required("vali_path")
  flags.mark_flag_as_required("test_path")
  flags.mark_flag_as_required("output_dir")

  tf.app.run()
