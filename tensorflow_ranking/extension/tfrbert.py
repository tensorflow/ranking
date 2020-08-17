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

"""Class defining utilities for finetuning TF-Ranking models with Bert."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.modeling import activations
from official.nlp import optimization
from official.nlp.bert import configs
from official.nlp.bert import tokenization
from official.nlp.modeling import networks as tfmodel_networks
from tensorflow_ranking.python.keras import network as tfrkeras_network
from tensorflow_serving.apis import input_pb2


class TFRBertRankingNetwork(tfrkeras_network.UnivariateRankingNetwork):
  """A TFRBertRankingNetwork scoring based univariate ranking network."""

  def __init__(self,
               context_feature_columns,
               example_feature_columns,
               bert_config_file,
               bert_max_seq_length,
               bert_output_dropout,
               name="tfrbert",
               **kwargs):
    """Initializes an instance of TFRBertRankingNetwork.

    Args:
      context_feature_columns: A dict containing all the context feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      example_feature_columns: A dict containing all the example feature columns
        used by the network. Keys are feature names, and values are instances of
        classes derived from `_FeatureColumn`.
      bert_config_file: (string) path to Bert configuration file.
      bert_max_seq_length: (int) maximum input sequence length (#words) after
        WordPiece tokenization. Sequences longer than this will be truncated,
        and shorter than this will be padded.
      bert_output_dropout: When not `None`, the probability will be used as the
        dropout probability for BERT output.
      name: name of Keras network.
      **kwargs: keyword arguments.
    """
    super(TFRBertRankingNetwork, self).__init__(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        name=name,
        **kwargs)

    self._bert_config_file = bert_config_file
    self._bert_max_seq_length = bert_max_seq_length
    self._bert_output_dropout = bert_output_dropout

    bert_config = configs.BertConfig.from_json_file(self._bert_config_file)
    self._bert_encoder = tfmodel_networks.TransformerEncoder(
        vocab_size=bert_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        activation=activations.gelu,
        dropout_rate=bert_config.hidden_dropout_prob,
        attention_dropout_rate=bert_config.attention_probs_dropout_prob,
        sequence_length=self._bert_max_seq_length,
        max_sequence_length=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range))

    self._dropout_layer = tf.keras.layers.Dropout(
        rate=self._bert_output_dropout)

    self._score_layer = tf.keras.layers.Dense(units=1, name="score")

  def score(self, context_features=None, example_features=None, training=True):
    """Univariate scoring of context and one example to generate a score.

    Args:
      context_features: (dict) Context feature names to 2D tensors of shape
        [batch_size, ...].
      example_features: (dict) Example feature names to 2D tensors of shape
        [batch_size, ...].
      training: (bool) Whether in training or inference mode.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, 1].
    """
    inputs = {
        "input_word_ids": tf.cast(example_features["input_ids"], tf.int32),
        "input_mask": tf.cast(example_features["input_mask"], tf.int32),
        "input_type_ids": tf.cast(example_features["segment_ids"], tf.int32)
    }

    # The `bert_encoder` returns a tuple of (sequence_output, cls_output).
    _, cls_output = self._bert_encoder(inputs, training=training)

    output = self._dropout_layer(cls_output, training=training)

    return self._score_layer(output)

  def get_config(self):
    config = super(TFRBertRankingNetwork, self).get_config()
    config.update({
        "bert_config_file": self._bert_config_file,
        "bert_max_seq_length": self._bert_max_seq_length,
        "bert_output_dropout": self._bert_output_dropout,
    })

    return config


class TFRBertUtil(object):
  """Class that defines a set of utility functions for Bert."""

  def __init__(self,
               bert_config_file,
               bert_init_ckpt,
               bert_max_seq_length,
               bert_vocab_file=None,
               do_lower_case=None):
    """Constructor.

    Args:
      bert_config_file: (string) path to Bert configuration file.
      bert_init_ckpt: (string)  path to pretrained Bert checkpoint.
      bert_max_seq_length: (int) maximum input sequence length (#words) after
        WordPiece tokenization. Sequences longer than this will be truncated,
        and shorter than this will be padded.
      bert_vocab_file (optional): (string) path to Bert vocabulary file.
      do_lower_case (optional): (bool) whether to lower case the input text.
        This should be aligned with the `vocab_file`.
    """
    self._bert_config_file = bert_config_file
    self._bert_init_ckpt = bert_init_ckpt
    self._bert_max_seq_length = bert_max_seq_length

    self._tokenizer = None
    if bert_vocab_file is not None and do_lower_case is not None:
      self._tokenizer = tokenization.FullTokenizer(
          vocab_file=bert_vocab_file, do_lower_case=do_lower_case)

  def create_optimizer(self,
                       init_lr,
                       train_steps,
                       warmup_steps,
                       optimizer_type="adamw"):
    """Creates an optimizer for TFR-BERT.

    Args:
      init_lr: (float) the init learning rate.
      train_steps: (int) the number of train steps.
      warmup_steps: (int) if global_step < num_warmup_steps, the learning rate
        will be `global_step / num_warmup_steps * init_lr`. See more details in
        the `tensorflow_models.official.nlp.optimization.py` file.
      optimizer_type: (string) Optimizer type, can either be `adamw` or `lamb`.
        Default to be the `adamw` (AdamWeightDecay). See more details in the
        `tensorflow_models.official.nlp.optimization.py` file.

    Returns:
      The optimizer training op.
    """

    return optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=train_steps,
        num_warmup_steps=warmup_steps,
        optimizer_type=optimizer_type)

  def get_warm_start_settings(self, exclude):
    """Defines warm-start settings for the TFRBert ranking estimator.

    Our TFRBert ranking models will warm-start from a pre-trained Bert model.
    Here, we define the warm-start setting by excluding non-Bert parameters.

    Args:
      exclude: (string) Variable to exclude from the warm-start settings.

    Returns:
      (`tf.estimator.WarmStartSettings`) the warm-start setting for the TFRBert
      ranking estimator.
    """
    # A regular expression to exclude the variables starting with the passed-in
    # `exclude` parameter. Variables from the downloaded Bert checkpoints often
    # start with `transformer`, `pooler`, `embeddings` and etc., whereas other
    # variables specifically to the TFRBertRankingNetwork start with the `name`
    # we passed to the `TFRBertRankingNetwork` constructor. When defining the
    # warm-start settings, we exclude those non-Bert variables.
    vars_to_warm_start = "^(?!{exclude}).*$".format(exclude=exclude)
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=self._bert_init_ckpt,
        vars_to_warm_start=vars_to_warm_start)

  def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which truncates the longer sequence until
    # their combined length is no longer than max_length.
    # This makes more sense than truncating with an equal percentage of tokens
    # from each, since if a sequence is very short then each token that's
    # truncated likely contains more information than a longer sequence.
    assert max_length > 0
    assert isinstance(max_length, int)
    if len(tokens_a) + len(tokens_b) > max_length:
      # Truncation is needed.
      if (len(tokens_a) >= max_length - max_length // 2 and
          len(tokens_b) >= max_length // 2):
        # Truncate both sequences until they have almost equal lengths and the
        # combined length is no longer than max_length
        del tokens_a[max_length - max_length // 2:]
        del tokens_b[max_length // 2:]
      elif len(tokens_a) > len(tokens_b):
        # Only truncating tokens_a would suffice
        del tokens_a[max_length - len(tokens_b):]
      else:
        # Only truncating tokens_b would suffice
        del tokens_b[max_length - len(tokens_a):]

  def _to_bert_ids(self, sent_a, sent_b=None):
    """Converts a sentence pair (sent_a, sent_b) to related Bert ids.

    This function is mostly adopted from run_classifier.convert_single_example
    in bert/run_classifier.py.

    Args:
      sent_a: (str) the raw text of the first sentence.
      sent_b: (str) the raw text of the second sentence.

    Returns:
      A tuple (`input_ids`, `input_masks`, `segment_ids`) for Bert finetuning.
    """

    if self._tokenizer is None:
      raise ValueError("Please pass both `vocab_file` and `do_lower_case` in "
                       "the BertUtil constructor to build a Bert tokenizer!")

    if sent_a is None:
      raise ValueError("`sent_a` cannot be None!")

    tokens_a = self._tokenizer.tokenize(sent_a)
    tokens_b = None if not sent_b else self._tokenizer.tokenize(sent_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total length is
      # less than the specified length. Since the final sequence will be
      # [CLS] `tokens_a` [SEP] `tokens_b` [SEP], thus, we use `- 3`.
      self._truncate_seq_pair(tokens_a, tokens_b, self._bert_max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2".  Since there is only one
      # sentence, we don't need to account for the second [SEP].
      self._truncate_seq_pair(tokens_a, [], self._bert_max_seq_length - 2)

    # The convention in BERT for sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    #
    # The `type_ids` (aka. `segment_ids`) are used to indicate whether this is
    # the first or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    #
    # When there is only one sentence given, the sequence pair would be:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] + [0] * len(tokens_a) + [0]
    if tokens_b:
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * len(tokens_b) + [1]
    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < self._bert_max_seq_length:
      padding_len = self._bert_max_seq_length - len(input_ids)
      input_ids.extend([0] * padding_len)
      input_mask.extend([0] * padding_len)
      segment_ids.extend([0] * padding_len)

    assert len(input_ids) == self._bert_max_seq_length
    assert len(input_mask) == self._bert_max_seq_length
    assert len(segment_ids) == self._bert_max_seq_length

    return input_ids, input_mask, segment_ids

  def convert_to_elwc(self, context, examples, labels, label_name):
    """Converts a <context, example list> pair to an ELWC example.

    Args:
      context: (str) raw text for a context (aka. query).
      examples: (list) raw texts for a list of examples (aka. documents).
      labels: (list) a list of labels (int) for the `examples`.
      label_name: (str) name of the label in the ELWC example.

    Returns:
      A tensorflow.serving.ExampleListWithContext example containing the
      `input_ids`, `input_masks`, `segment_ids` and `label_id` fields.
    """
    if len(examples) != len(labels):
      raise ValueError("`examples` and `labels` should have the same size!")

    elwc = input_pb2.ExampleListWithContext()
    for example, label in zip(examples, labels):
      (input_ids, input_mask, segment_ids) = self._to_bert_ids(context, example)

      feature = {
          "input_ids":
              tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
          "input_mask":
              tf.train.Feature(int64_list=tf.train.Int64List(value=input_mask)),
          "segment_ids":
              tf.train.Feature(
                  int64_list=tf.train.Int64List(value=segment_ids)),
          label_name:
              tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
      }
      tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
      elwc.examples.append(tf_example)

    return elwc
