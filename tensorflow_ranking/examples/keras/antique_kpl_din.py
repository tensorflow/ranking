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
r"""Antique TFR-KPL trainer for Document Interaction Network.

--------------------------------------------------------------------------------
Sample command lines:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_elwc.tfrecord && \
EVAL=tensorflow_ranking/examples/data/eval_elwc.tfrecord && \
VOCAB=tensorflow_ranking/examples/data/vocab.txt && \
rm -rf $MODEL_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/keras/antique_kpl_din && \
./bazel-bin/tensorflow_ranking/examples/keras/antique_kpl_din \
--train_file_pattern=$TRAIN \
--eval_file_pattern=$EVAL \
--vocab_file_path=$VOCAB \
--model_dir=$MODEL_DIR

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
"""
import os
from typing import Dict, Optional, Tuple, Union

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_string("train_input_pattern", None,
                    "Input file path pattern used for training.")

flags.DEFINE_string("eval_input_pattern", None,
                    "Input file path pattern used for eval.")

flags.DEFINE_string("test_input_pattern", None,
                    "Input file path pattern used for test.")

flags.DEFINE_string(
    "vocab_file_path", None, "Path to vocab file used for tokenizing the "
    "Antique dataset.")

flags.DEFINE_integer(
    "vocab_size", 30522, "Size of the vocab file used for "
    "tokenizing the Antique dataset.")

flags.DEFINE_float("learning_rate", 0.005, "Learning rate for optimizer.")

flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")

flags.DEFINE_bool("use_batch_norm", True, "Whether to use batch normalization.")

flags.DEFINE_float("batch_normalization_moment", 0.99,
                   "Batch Normalization moment parameter.")

flags.DEFINE_list("hidden_layer_dims", None,
                  "Number of units in each hidden layer.")

flags.DEFINE_integer("train_batch_size", 16,
                     "Number of input records used per batch for training.")

flags.DEFINE_integer("eval_batch_size", 64,
                     "Number of input records used per batch for eval.")

flags.DEFINE_integer("num_train_steps", 15000, "Number of training iterations.")

flags.DEFINE_integer("num_epochs", 100,
                     "Number of passes over the training data.")

flags.DEFINE_integer("list_size", 50, "List size used for training.")

flags.DEFINE_string("loss", "approx_ndcg_loss",
                    "See tfr.losses.RankingLossKey.")

flags.DEFINE_integer("embedding_dimension", 20, "Size of embedding.")

flags.DEFINE_string(
    "model_dir", None, "The directory where the model weights and "
    "training/evaluation summaries are stored.")

flags.DEFINE_bool(
    "use_document_interaction", True,
    "If True, use Document Interaction Network to capture cross-document "
    "interactions as additional features for scoring.")

flags.DEFINE_integer(
    "num_attention_layers", 1, "number of attention layers. See "
    "`tfr.keras.layers.DocumentInteractionAttention`.")

flags.DEFINE_integer(
    "num_attention_heads", 1, "number of self attention heads. See "
    "`tfr.keras.layers.DocumentInteractionAttention`.")

flags.DEFINE_integer(
    "head_size", 128, "Size of attention head. See "
    "`tfr.keras.layers.DocumentInteractionAttention`.")

FLAGS = flags.FLAGS

# The document relevance label.
_LABEL_FEATURE = "relevance"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1.
_MASK = "example_list_mask"

# Context and example features.
_CONTEXT_RAGGED_FEATURE_KEYS = ["query_tokens"]
_EXAMPLE_RAGGED_FEATURE_KEYS = ["document_tokens"]


def _create_feature_spec(
    with_label: Optional[bool] = False
) -> Tuple[Dict[str, Union[tf.io.FixedLenFeature, tf.io.RaggedFeature]], Dict[
    str, Union[tf.io.FixedLenFeature, tf.io.RaggedFeature]]]:
  """Create context and example feature spec for data parsing.

  Args:
      with_label: Includes the label spec if True. Set this to True for
        training.

  Returns:
    A pair of dicts for context and example features, mapping each feature key
      to a FixedLenFeature or RaggedFeature.
  """
  context_feature_spec = {
      feature_key: tf.io.RaggedFeature(dtype=tf.string)
      for feature_key in _CONTEXT_RAGGED_FEATURE_KEYS
  }
  example_feature_spec = {
      feature_key: tf.io.RaggedFeature(dtype=tf.string)
      for feature_key in _EXAMPLE_RAGGED_FEATURE_KEYS
  }
  if with_label:
    example_feature_spec[_LABEL_FEATURE] = tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.int64, default_value=(int(_PADDING_LABEL),))
  return context_feature_spec, example_feature_spec


def _create_dataset(file_pattern: str,
                    batch_size: int,
                    randomize_input: bool = True,
                    num_epochs: Optional[int] = None,
                    drop_final_batch: bool = False,
                    reader_num_threads: int = 10) -> tf.data.Dataset:
  """Returns `tf.data.Dataset` for training or evaluating the model.

  Args:
    file_pattern: (string) file pattern for input data.
    batch_size: (int) number of input examples to process per batch.
    randomize_input: (bool) if true, randomize input example order. It should
      almost always be true except for unittest/debug purposes.
    num_epochs: (int) Number of times the input dataset must be repeated. None
      to repeat the data indefinitely.
    drop_final_batch: (boolean) Whether to drop the last batch if there are
      fewer than batch_size elements left.
    reader_num_threads: (int) Number of parallel threads to read from.

  Returns:
    A `tf.data.Dataset` of features dictionary and label.
  """
  context_feature_spec, example_feature_spec = _create_feature_spec(
      with_label=True)
  dataset = tfr.data.build_ranking_dataset(
      data_format=tfr.data.ELWC,
      file_pattern=file_pattern,
      batch_size=batch_size,
      list_size=FLAGS.list_size,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      reader=tf.data.TFRecordDataset,
      num_epochs=num_epochs,
      shuffle=randomize_input,
      reader_num_threads=reader_num_threads,
      drop_final_batch=drop_final_batch,
      mask_feature_name=_MASK)

  def _add_label(features):
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, dtype=tf.float32)
    return features, label

  dataset = dataset.map(_add_label)
  return dataset


def create_keras_inputs(
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor]:
  """Create Keras Input placeholder tensors.

  Returns:
    A tuple of two dicts mapping context and example feature names to
    tf.keras.Input.
  """
  context_keras_inputs, example_keras_inputs = {}, {}
  # Create Keras inputs for context features.
  context_keras_inputs.update({
      k: tf.keras.Input(name=k, shape=(None,), dtype=tf.string, ragged=True)
      for k in _CONTEXT_RAGGED_FEATURE_KEYS
  })
  # Create Keras inputs for example features.
  # Setting feature dimension to be None, so that variable number of tokens can
  # be processed for embedding. Note that the embeddings are reduced via a
  # SUM, MEAN or user defined reduction for a fixed feature dimension which is
  # equal to the embedding size.
  example_keras_inputs.update({
      k:
      tf.keras.Input(name=k, shape=(None, None), dtype=tf.string, ragged=True)
      for k in _EXAMPLE_RAGGED_FEATURE_KEYS
  })
  mask = tf.keras.Input(name=_MASK, shape=(None,), dtype=tf.bool)

  return context_keras_inputs, example_keras_inputs, mask


def preprocess_keras_inputs(
    context_keras_inputs: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
    example_keras_inputs: Dict[str, Union[tf.Tensor, tf.RaggedTensor]],
    mask: tf.Tensor
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
  """Preprocess context and example inputs.

  Args:
    context_keras_inputs: Mapping context feature names to tf.keras.Input.
    example_keras_inputs: Mapping example feature names to tf.keras.Input.
    mask: 2-D mask for valid examples of shape [batch_size, list_size].

  Returns:
    preprocessed_context_features: Mapping from context feature names
      to preprocessed dense 2-D tensors of shape [batch_size, ...].
    preprocessed_example_features: Mapping from example feature names
      to preprocessed dense 3-D tensors of shape [batch_size, list_size, ...].
  """
  ragged_to_dense_layer = tf.keras.layers.Lambda(
      lambda x: x.to_tensor() if isinstance(x, tf.RaggedTensor) else x)

  # Create shared embedding layer for context and example features.
  shared_embedding_layer = tf.keras.layers.Embedding(
      input_dim=FLAGS.vocab_size,
      output_dim=FLAGS.embedding_dimension,
      embeddings_initializer=None,
      embeddings_constraint=None)
  id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
      max_tokens=FLAGS.vocab_size,
      vocabulary=FLAGS.vocab_file_path,
      num_oov_indices=0,
      oov_token="[UNK#]",
      mask_token=None)

  def _embedding(x):
    ids = id_lookup_layer(x)
    embedded_tensor = shared_embedding_layer(ids)
    # Using mean reduction here. User can specify any kind of reduction,
    # e.g. sum. The reduction is along the ragged axis for variable number of
    # tokens. This reduction transforms a ragged feature of dense shape:
    # [batch_size, list_size, num_tokens, embedding_size] ->
    # [batch_size, list_size, embedding_size].
    embedded_tensor = tf.reduce_mean(embedded_tensor, axis=-2)
    return ragged_to_dense_layer(embedded_tensor)

  preprocessed_context_features, preprocessed_example_features = {}, {}
  preprocessed_context_features.update({
      k: _embedding(context_keras_inputs[k])
      for k in _CONTEXT_RAGGED_FEATURE_KEYS
  })
  preprocessed_example_features.update({
      k: _embedding(example_keras_inputs[k])
      for k in _EXAMPLE_RAGGED_FEATURE_KEYS
  })

  # Document interaction attention layer.
  if FLAGS.use_document_interaction:
    concat_tensor = tfr.keras.layers.ConcatFeatures()(
        inputs=(preprocessed_context_features, preprocessed_example_features,
                mask))
    din_layer = tfr.keras.layers.DocumentInteractionAttention(
        num_heads=FLAGS.num_attention_heads,
        head_size=FLAGS.head_size,
        num_layers=FLAGS.num_attention_layers,
        dropout=FLAGS.dropout_rate)
    preprocessed_example_features["document_interaction_embedding"] = din_layer(
        inputs=(concat_tensor, mask))

  return preprocessed_context_features, preprocessed_example_features


def create_ranking_model() -> tf.keras.Model:
  """Create ranking model using Functional API."""
  context_keras_inputs, example_keras_inputs, mask = create_keras_inputs()
  context_features, example_features = preprocess_keras_inputs(
      context_keras_inputs, example_keras_inputs, mask)

  (flattened_context_features,
   flattened_example_features) = tfr.keras.layers.FlattenList()(
       inputs=(context_features, example_features, mask))

  # Concatenate flattened context and example features along `list_size` dim.
  context_input = [
      tf.keras.layers.Flatten()(flattened_context_features[name])
      for name in sorted(flattened_context_features)
  ]
  example_input = [
      tf.keras.layers.Flatten()(flattened_example_features[name])
      for name in sorted(flattened_example_features)
  ]
  input_layer = tf.concat(context_input + example_input, 1)

  # User can create a custom scoring logic as a sequence of layers.
  dnn = tf.keras.Sequential()
  # Input batch normalization.
  if FLAGS.use_batch_norm:
    dnn.add(
        tf.keras.layers.BatchNormalization(
            momentum=FLAGS.batch_normalization_moment))
  for layer_size in FLAGS.hidden_layer_dims:
    dnn.add(tf.keras.layers.Dense(units=layer_size))
    if FLAGS.use_batch_norm:
      dnn.add(
          tf.keras.layers.BatchNormalization(
              momentum=FLAGS.batch_normalization_moment))
    dnn.add(tf.keras.layers.Activation(activation=tf.nn.relu))
    dnn.add(tf.keras.layers.Dropout(rate=FLAGS.dropout_rate))
  dnn.add(tf.keras.layers.Dense(units=1))

  logits = tfr.keras.layers.RestoreList()(inputs=(dnn(input_layer), mask))

  return tf.keras.Model(
      inputs=dict(
          list(context_keras_inputs.items()) +
          list(example_keras_inputs.items()) + [(_MASK, mask)]),
      outputs=logits,
      name="din_ranking_model")


def train_and_eval():
  """Train and evaluate ranking model."""
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=FLAGS.learning_rate)
  loss = tfr.keras.losses.get(loss=FLAGS.loss)
  eval_metrics = tfr.keras.metrics.default_keras_metrics()

  model = create_ranking_model()
  model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)
  model.summary()

  tensorboard_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir)

  train_dataset = _create_dataset(
      file_pattern=FLAGS.train_input_pattern, batch_size=FLAGS.train_batch_size)

  eval_dataset = _create_dataset(
      file_pattern=FLAGS.eval_input_pattern,
      batch_size=FLAGS.eval_batch_size,
      num_epochs=1,
      randomize_input=True)

  # Train ranker.
  steps_per_epoch = FLAGS.num_train_steps // FLAGS.num_epochs
  logging.info("Training the model...")
  model.fit(
      train_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=FLAGS.num_epochs,
      validation_data=eval_dataset,
      callbacks=[tensorboard_callback])
  logging.info("Finished training the model.")

  # Evaluate over test data.
  if FLAGS.test_input_pattern is not None:
    logging.info("Evaluating the model...")
    test_dataset = _create_dataset(
        file_pattern=FLAGS.test_input_pattern,
        batch_size=FLAGS.eval_batch_size,
        randomize_input=False,
        num_epochs=1)
    model.evaluate(test_dataset)
    logging.info("Finished evaluating the model.")

  # Export SavedModel.
  context_feature_spec, example_feature_spec = _create_feature_spec(
      with_label=False)
  saved_model_path = os.path.join(FLAGS.model_dir, "export/")
  logging.info("Exporting to SavedModel...")
  model.save(
      filepath=saved_model_path,
      signatures=tfr.keras.saved_model.Signatures(
          model,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          mask_feature_name=_MASK)())
  logging.info("SavedModel exported successfully to: %s", saved_model_path)
  return model


def main(_):
  tf.random.set_seed(1234)
  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_input_pattern")
  flags.mark_flag_as_required("eval_input_pattern")
  flags.mark_flag_as_required("vocab_file_path")
  app.run(main)
