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
r"""Antique TFR-KPL trainer using ragged tensors.

This file has two ways to construct and train a model. One is using the standard
way of `tf.keras.Model`. The other is based on the `tfr.keras.pipeline` and
`tfr.keras.model`. The two are controlled by FLAGS.use_pipeline.

--------------------------------------------------------------------------------
Sample command lines:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_elwc.tfrecord && \
EVAL=tensorflow_ranking/examples/data/eval_elwc.tfrecord && \
VOCAB=tensorflow_ranking/examples/data/vocab.txt && \
rm -rf $MODEL_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/keras/antique_ragged && \
./bazel-bin/tensorflow_ranking/examples/keras/antique_ragged \
--train_file_pattern=$TRAIN \
--eval_file_pattern=$EVAL \
--vocab_file_path=$VOCAB \
--model_dir=$MODEL_DIR

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
  * Use --use_pipeline to use the `tfr.keras.pipeline`.
"""
import os
from typing import List

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import tensorflow_ranking as tfr

# Data config.
flags.DEFINE_string("train_input_pattern", None,
                    "Input file path pattern used for training.")
flags.DEFINE_string("eval_input_pattern", None,
                    "Input file path pattern used for eval.")
flags.DEFINE_string("test_input_pattern", None,
                    "Input file path pattern used for test.")

# Model config.
flags.DEFINE_string(
    "vocab_file_path", None, "Path to vocab file used for tokenizing the "
    "Antique dataset.")
flags.DEFINE_integer(
    "vocab_size", 30522, "Size of the vocab file used for "
    "tokenizing the Antique dataset.")
flags.DEFINE_integer("embedding_dimension", 20, "Size of embedding.")
flags.DEFINE_multi_integer("hidden_layer_dims", [20, 10],
                           "Number of units in each hidden layer.")

# Training config.
flags.DEFINE_string("loss", tfr.keras.losses.RankingLossKey.APPROX_NDCG_LOSS,
                    "See tfr.keras.losses.RankingLossKey.")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate for optimizer.")
flags.DEFINE_integer("train_batch_size", 16,
                     "Number of input records used per batch for training.")
flags.DEFINE_integer("eval_batch_size", 64,
                     "Number of input records used per batch for eval.")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of passes over the training data.")
flags.DEFINE_string(
    "model_dir", None, "The directory where the model weights and "
    "training/evaluation summaries are stored.")
flags.DEFINE_integer("num_train_steps", 100000,
                     "Number of training iterations.")
flags.DEFINE_integer("num_valid_steps", 100, "Number of validation iterations.")

# Others.
flags.DEFINE_bool("use_pipeline", False,
                  "If True, use the pipeline for training.")

FLAGS = flags.FLAGS

# The document relevance label and mask feature.
_LABEL_FEATURE = "relevance"
_MASK = "example_list_mask"


class AntiqueEmbeddingRankingModel(tf.keras.Model):
  """A text-embedding ranking model for the Antique dataset.

  This class implements a ranking model that uses word embeddings followed by a
  feedforward neural network to produce document scores.
  """

  def __init__(self, vocab_size: int, vocab_file_path: str, embedding_dim: int,
               hidden_dims: List[int]):
    super().__init__()
    self._lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
        max_tokens=vocab_size,
        vocabulary=vocab_file_path,
        num_oov_indices=10,
        mask_token=None)

    self._embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size + 10,
        output_dim=embedding_dim,
        embeddings_initializer=None,
        embeddings_constraint=None)

    tower = tf.keras.Sequential([tf.keras.layers.BatchNormalization()])
    for hidden_dim in hidden_dims:
      tower.add(tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu))
    tower.add(tf.keras.layers.Dense(1))

    # A `TimeDistributed` layer will apply the tower model to each entry in the
    # second dimension of the input tensor independently and handles ragged
    # tensors. This makes it possible to compute a score for each document.
    self._ragged_tower = tf.keras.layers.TimeDistributed(tower)

  def call(self, inputs):
    # Get query and document tokens.
    query_tokens, doc_tokens = inputs["query_tokens"], inputs["document_tokens"]

    # Embed query and document tokens via a shared embedding layer.
    query_token_ids = self._lookup(query_tokens)
    doc_token_ids = self._lookup(doc_tokens)
    query_embeddings = self._embedding(query_token_ids)
    doc_embeddings = self._embedding(doc_token_ids)

    # Compute average embedding by averaging the token embeddings.
    query_embeddings = tf.reduce_mean(query_embeddings, axis=1)
    doc_embeddings = tf.reduce_mean(doc_embeddings, axis=2)

    # Broadcast query embedding over all documents and concatenate embeddings.
    query_embeddings = tf.ones_like(doc_embeddings) * tf.expand_dims(
        query_embeddings, axis=1)
    final_embeddings = tf.concat([query_embeddings, doc_embeddings], axis=2)

    # Run concatenated embeddings through a feed-forward scoring tower and
    # return the result.
    scores = self._ragged_tower(final_embeddings)
    return tf.squeeze(scores, axis=2)


def _add_ragged_label(inputs):
  mask = inputs[_MASK]
  features_dict = {
      "query_tokens": inputs["query_tokens"],
      "document_tokens": inputs["document_tokens"]
  }
  label = tf.ragged.boolean_mask(inputs[_LABEL_FEATURE], mask)
  label = tf.cast(label, dtype=tf.float32)
  return features_dict, label


def datasets():
  """Creates the datasets."""
  # Create feature specification.
  context_feature_spec = {
      "query_tokens": tf.io.RaggedFeature(dtype=tf.string),
  }
  example_feature_spec = {
      "document_tokens":
          tf.io.RaggedFeature(dtype=tf.string),
      _LABEL_FEATURE:
          tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
  }

  # Load datasets.
  train_dataset = tfr.data.build_ranking_dataset(
      file_pattern=FLAGS.train_input_pattern,
      data_format=tfr.data.ELWC,
      batch_size=FLAGS.train_batch_size,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      mask_feature_name=_MASK,
      num_epochs=1,
      shuffle_buffer_size=1000)
  eval_dataset = tfr.data.build_ranking_dataset(
      file_pattern=FLAGS.eval_input_pattern,
      data_format=tfr.data.ELWC,
      batch_size=FLAGS.eval_batch_size,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      mask_feature_name=_MASK,
      num_epochs=1,
      shuffle_buffer_size=1000)

  # This maps the dataset features to a tuple (features, label) where the label
  # is converted to a ragged tensor.
  train_dataset = train_dataset.map(_add_ragged_label)
  eval_dataset = eval_dataset.map(_add_ragged_label)
  return train_dataset, eval_dataset


def standalone_train_and_eval():
  """Train and evaluate ranking model."""
  train_dataset, eval_dataset = datasets()

  # Create optimizer, ranking loss and ranking metrics.
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=FLAGS.learning_rate)
  loss = tfr.keras.losses.get(loss=FLAGS.loss, ragged=True)
  eval_metrics = tfr.keras.metrics.default_keras_metrics(ragged=True)

  # Create ranking model to train. This model operates on ragged tensors and
  # returns model scores as ragged tensors.
  model = AntiqueEmbeddingRankingModel(
      vocab_size=FLAGS.vocab_size,
      vocab_file_path=FLAGS.vocab_file_path,
      embedding_dim=FLAGS.embedding_dimension,
      hidden_dims=FLAGS.hidden_layer_dims)
  model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)

  # Train ranker.
  logging.info("Training the model...")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir)
  model.fit(
      train_dataset,
      epochs=FLAGS.num_epochs,
      steps_per_epoch=FLAGS.num_train_steps // FLAGS.num_epochs,
      validation_data=eval_dataset,
      validation_steps=FLAGS.num_valid_steps,
      callbacks=[tensorboard_callback])
  logging.info("Finished training the model.")

  # Save model.
  saved_model_path = os.path.join(FLAGS.model_dir, "export/")
  logging.info("Exporting to SavedModel...")
  model.save(filepath=saved_model_path)
  logging.info("SavedModel exported successfully to: %s", saved_model_path)


########
# The following is to support training with tfr.keras.pipeline.
########
class MyModelBuilder(tfr.keras.model.AbstractModelBuilder):
  """Wraps the model into a ModelBuilder to work with `tfr.keras.pipeline`."""

  def build(self) -> tf.keras.Model:
    """Builds the model."""
    return AntiqueEmbeddingRankingModel(
        vocab_size=FLAGS.vocab_size,
        vocab_file_path=FLAGS.vocab_file_path,
        embedding_dim=FLAGS.embedding_dimension,
        hidden_dims=FLAGS.hidden_layer_dims)


class RaggedPipeline(tfr.keras.pipeline.SimplePipeline):
  """Supports ragged tensors."""

  def build_loss(self):
    """Builds the loss for ragged."""
    return tfr.keras.losses.get(loss=FLAGS.loss, ragged=True)

  def build_metrics(self):
    """Builds the metrics for ragged."""
    return tfr.keras.metrics.default_keras_metrics(ragged=True)


def pipeline_train_and_eval():
  """Train and evaluate ranking model."""
  train_dataset, eval_dataset = datasets()
  pipeline_hparams = tfr.keras.pipeline.PipelineHparams(
      model_dir=FLAGS.model_dir,
      num_epochs=FLAGS.num_epochs,
      steps_per_epoch=(FLAGS.num_train_steps // FLAGS.num_epochs),
      validation_steps=FLAGS.num_valid_steps,
      loss=FLAGS.loss,
      loss_reduction=tf.losses.Reduction.AUTO,
      optimizer="adagrad",
      learning_rate=FLAGS.learning_rate,
      steps_per_execution=10,
      export_best_model=True,
      strategy="MirroredStrategy")

  ranking_pipeline = RaggedPipeline(
      model_builder=MyModelBuilder(),
      dataset_builder=tfr.keras.pipeline.NullDatasetBuilder(
          train_dataset, eval_dataset),
      hparams=pipeline_hparams)
  ranking_pipeline.train_and_validate()


def main(_):
  tf.random.set_seed(1234)
  if FLAGS.use_pipeline:
    pipeline_train_and_eval()
  else:
    standalone_train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_input_pattern")
  flags.mark_flag_as_required("eval_input_pattern")
  flags.mark_flag_as_required("vocab_file_path")
  app.run(main)
