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

"""Provides a `RankingPipeline` for running a TF-Ranking model.

This class contains the boilerplate required to run a TF-Ranking model, which
reduces a few replicated setups (e.g., input function, serving input function,
model export strategies) for running TF-Ranking models. Advanced users can also
derive from this class and further tailor for their needs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_ranking.python import data as tfr_data

_PADDING_LABEL = -1


class RankingPipeline(object):
  """Class to set up the input, train and eval processes for a TF Ranking model.

  An example use case is provided below:

  ```python
  import tensorflow as tf
  import tensorflow_ranking as tfr

  context_feature_columns = {
    "c1": tf.feature_column.numeric_column("c1", shape=(1,))
  }
  example_feature_columns = {
    "e1": tf.feature_column.numeric_column("e1", shape=(1,))
  }

  hparams = dict(
        train_input_pattern="/path/to/train/files",
        eval_input_pattern="/path/to/eval/files",
        train_batch_size=8,
        eval_batch_size=8,
        checkpoint_secs=120,
        num_checkpoints=1000,
        num_train_steps=10000,
        num_eval_steps=100,
        loss="softmax_loss",
        list_size=10,
        listwise_inference=False,
        convert_labels_to_binary=False,
        model_dir="/path/to/your/model/directory")

  # See `tensorflow_ranking.estimator` for details about creating an estimator.
  estimator = <create your own estimator>

  ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
        context_feature_columns,
        example_feature_columns,
        hparams,
        estimator=estimator,
        label_feature_name="relevance",
        label_feature_type=tf.int64)
  ranking_pipeline.train_and_eval()
  ```

  Note that you may:
    * pass `best_exporter_metric` and `best_exporter_metric_higher_better` for
      different model export strategies.
    * pass `dataset_reader` for reading different `tf.Dataset`s. We recommend
      using TFRecord files and storing your data in `tfr.data.ELWC` format.

  If you want to further customize certain `RankingPipeline` behaviors, please
  create a subclass of `RankingPipeline`, and overwrite related functions. We
  recommend only overwriting the following functions:
    * `_make_dataset` which builds the tf.dataset for a tf-ranking model.
    * `_make_serving_input_fn` that defines the input function for serving.
    * `_export_strategies` if you have more advanced needs for model exporting.

  For example, if you want to remove the best exporters, you may overwrite:

  ```python
  class NoBestExporterRankingPipeline(tfr.ext.pipeline.RankingPipeline):
    def _export_strategies(self, event_file_pattern):
      del event_file_pattern
      latest_exporter = tf.estimator.LatestExporter(
          "latest_model",
          serving_input_receiver_fn=self._make_serving_input_fn())
      return [latest_exporter]

  ranking_pipeline = NoBestExporterRankingPipeline(
        context_feature_columns,
        example_feature_columns,
        hparams
        estimator=estimator)
  ranking_pipeline.train_and_eval()
  ```

  if you want to customize your dataset reading behaviors, you may overwrite:

  ```python
  class CustomizedDatasetRankingPipeline(tfr.ext.pipeline.RankingPipeline):
    def _make_dataset(self,
                      batch_size,
                      list_size,
                      input_pattern,
                      randomize_input=True,
                      num_epochs=None):
      # Creates your own dataset, plese follow `tfr.data.build_ranking_dataset`.
      dataset = build_my_own_ranking_dataset(...)
      ...
      return dataset.map(self._features_and_labels)

  ranking_pipeline = CustomizedDatasetRankingPipeline(
        context_feature_columns,
        example_feature_columns,
        hparams
        estimator=estimator)
  ranking_pipeline.train_and_eval()
  ```

  """

  def __init__(self,
               context_feature_columns,
               example_feature_columns,
               hparams,
               estimator,
               label_feature_name="relevance",
               label_feature_type=tf.int64,
               dataset_reader=tf.data.TFRecordDataset,
               best_exporter_metric=None,
               best_exporter_metric_higher_better=True,
               size_feature_name=None):
    """Constructor.

    Args:
      context_feature_columns: (dict) Context (aka, query) feature columns.
      example_feature_columns: (dict) Example (aka, document) feature columns.
      hparams: (dict) A dict containing model hyperparameters.
      estimator: (`Estimator`) An `Estimator` instance for model train and eval.
      label_feature_name: (str) The name of the label feature.
      label_feature_type: (`tf.dtype`) The value type of the label feature.
      dataset_reader: (`tf.Dataset`) The dataset format for the input files.
      best_exporter_metric: (str) Metric key for exporting the best model. If
        None, exports the model with the minimal loss value.
      best_exporter_metric_higher_better: (bool) If a higher metric is better.
        This is only used if `best_exporter_metric` is not None.
      size_feature_name: (str) If set, populates the feature dictionary with
        this name and the coresponding value is a `tf.int32` Tensor of shape
        [batch_size] indicating the actual sizes of the example lists before
        padding and truncation. If None, which is default, this feature is not
        generated.
    """
    self._validate_parameters(estimator, hparams)

    self._context_feature_columns = context_feature_columns
    self._example_feature_columns = example_feature_columns
    self._hparams = hparams
    self._estimator = estimator
    self._label_feature_name = label_feature_name
    self._label_feature_type = label_feature_type
    self._dataset_reader = dataset_reader
    self._best_exporter_metric = best_exporter_metric
    self._best_exporter_metric_higher_better = (
        best_exporter_metric_higher_better)
    self._size_feature_name = size_feature_name

  def _required_hparam_keys(self):
    """Returns a list of keys for the required hparams for RankingPipeline."""
    required_hparam_keys = [
        "train_input_pattern", "eval_input_pattern", "train_batch_size",
        "eval_batch_size", "checkpoint_secs", "num_checkpoints",
        "num_train_steps", "num_eval_steps", "loss", "list_size",
        "convert_labels_to_binary", "model_dir", "listwise_inference"
    ]
    return required_hparam_keys

  def _validate_parameters(self, estimator, hparams):
    """Validates the passed-in estimator and hparams.

    Args:
      estimator: (`Estimator`) An `Estimator` instance.
      hparams: (dict) A dict containing model hyperparameters.

    Raises:
      ValueError: If the `estimator` is None.
      ValueError: If the `estimator` is not an `Estimator`.
      ValueError: If any of the `self._required_hparam_keys()` does not present
        in the `hparams`.
    """
    if estimator is None:
      raise ValueError("The `estimator` cannot be empty!")

    if not isinstance(
        estimator, (tf.estimator.Estimator, tf.compat.v1.estimator.Estimator)):
      raise ValueError(
          "The argument estimator needs to be of type tf.estimator.Estimator, "
          "not %s." % type(estimator))

    for required_key in self._required_hparam_keys():
      if required_key not in hparams:
        raise ValueError("Required key is missing: '{}'".format(required_key))

  def _features_and_labels(self, features):
    """Extracts labels from features."""
    label = tf.cast(
        tf.squeeze(features.pop(self._label_feature_name), axis=2), tf.float32)
    if self._hparams.get("convert_labels_to_binary"):
      label = tf.compat.v1.where(
          tf.greater(label, 0.), tf.ones_like(label), label)
    return features, label

  def _make_dataset(self,
                    batch_size,
                    list_size,
                    input_pattern,
                    randomize_input=True,
                    num_epochs=None):
    """Builds a dataset for the TF-Ranking model.

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

    dataset = tfr_data.build_ranking_dataset(
        file_pattern=input_pattern,
        data_format=tfr_data.ELWC,
        batch_size=batch_size,
        list_size=list_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=self._dataset_reader,
        reader_args=None,
        num_epochs=num_epochs,
        shuffle=randomize_input,
        shuffle_buffer_size=1000,
        shuffle_seed=None,
        prefetch_buffer_size=10000,
        reader_num_threads=64,
        sloppy_ordering=True,
        drop_final_batch=False,
        num_parser_threads=None,
        size_feature_name=self._size_feature_name)

    return dataset.map(self._features_and_labels)

  def _make_input_fn(self,
                     input_pattern,
                     batch_size,
                     list_size,
                     randomize_input=True,
                     num_epochs=None):
    """Returns the input function for the ranking model.

    Args:
      input_pattern: (str) File pattern for the input data.
      batch_size: (int) The number of input examples to process per batch.
      list_size: (int) The list size for an ELWC example.
      randomize_input: (bool) If true, randomize input example order. It should
        almost always be true except for unittest/debug purposes.
      num_epochs: (int) The number of times the input dataset must be repeated.
        None to repeat the data indefinitely.

    Returns:
      An `input_fn` for `tf.estimator.Estimator`.
    """

    def _input_fn():
      """`input_fn` for the `Estimator`."""
      return self._make_dataset(
          batch_size=batch_size,
          list_size=list_size,
          input_pattern=input_pattern,
          randomize_input=randomize_input,
          num_epochs=num_epochs)

    return _input_fn

  def _make_serving_input_fn(self):
    """Returns `Estimator` `input_fn` for serving the model.

    Returns:
      `input_fn` that can be used in serving. The returned input_fn takes no
      arguments and returns `InputFnOps'.
    """
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        self._context_feature_columns.values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        self._example_feature_columns.values())

    if self._hparams.get("listwise_inference"):
      # Exports accept the `ExampleListWithContext` format during serving.
      return tfr_data.build_ranking_serving_input_receiver_fn(
          data_format=tfr_data.ELWC,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          size_feature_name=self._size_feature_name)
    else:
      # Exports accept `tf.Example` format during serving.
      feature_spec = {}
      feature_spec.update(example_feature_spec)
      feature_spec.update(context_feature_spec)
      return tf.estimator.export.build_parsing_serving_input_receiver_fn(
          feature_spec)

  def _export_strategies(self, event_file_pattern):
    """Defines the export strategies.

    Args:
      event_file_pattern: (str) Event file name pattern relative to model_dir.

    Returns:
      A list of `tf.Exporter` strategies for model exporting.
    """
    export_strategies = []

    latest_exporter = tf.estimator.LatestExporter(
        "latest_model", serving_input_receiver_fn=self._make_serving_input_fn())
    export_strategies.append(latest_exporter)

    # In case of not specifying the `best_exporter_metric`, uses the default
    # BestExporter by the loss value.
    if self._best_exporter_metric is None:
      best_exporter = tf.estimator.BestExporter(
          name="best_model_by_loss",
          serving_input_receiver_fn=self._make_serving_input_fn(),
          event_file_pattern=event_file_pattern)
      export_strategies.append(best_exporter)
      return export_strategies

    def _compare_fn(best_eval_result, current_eval_result):
      """A `compare_fn` to determine the best evaluation result."""
      if self._best_exporter_metric not in current_eval_result:
        raise ValueError(
            "Metric `%s` does not exist! Please use any of the following: `%s`."
            % (self._best_exporter_metric, current_eval_result.keys()))

      is_current_the_best = (
          self._best_exporter_metric_higher_better == (
              current_eval_result[self._best_exporter_metric] >=
              best_eval_result[self._best_exporter_metric]))

      return is_current_the_best

    best_exporter = tf.estimator.BestExporter(
        name="best_model_by_metric",
        serving_input_receiver_fn=self._make_serving_input_fn(),
        event_file_pattern=event_file_pattern,
        compare_fn=_compare_fn)
    export_strategies.append(best_exporter)

    return export_strategies

  def _train_eval_specs(self):
    """Makes a tuple of (train_spec, eval_on_eval_spec, eval_on_train_spec)."""
    train_list_size = self._hparams.get("list_size")
    eval_list_size = self._hparams.get("eval_list_size") or train_list_size
    train_input_fn = self._make_input_fn(
        input_pattern=self._hparams.get("train_input_pattern"),
        batch_size=self._hparams.get("train_batch_size"),
        list_size=train_list_size)
    eval_input_fn = self._make_input_fn(
        input_pattern=self._hparams.get("eval_input_pattern"),
        batch_size=self._hparams.get("eval_batch_size"),
        list_size=eval_list_size)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=self._hparams.get("num_train_steps"))
    eval_on_train_spec = tf.estimator.EvalSpec(
        name="on_train",
        input_fn=train_input_fn,
        steps=self._hparams.get("num_eval_steps"),
        throttle_secs=5)

    eval_on_eval_spec = tf.estimator.EvalSpec(
        name="on_eval",
        input_fn=eval_input_fn,
        steps=self._hparams.get("num_eval_steps"),
        exporters=self._export_strategies(
            event_file_pattern="eval_on_eval/*.tfevents.*"),
        throttle_secs=5)
    return train_spec, eval_on_eval_spec, eval_on_train_spec

  def train_and_eval(self, local_training=True):
    """Launches train and evaluation jobs locally."""
    # TODO: supports for distributed training and evaluation.
    if not local_training:
      raise ValueError("The non local training is not supported now!")

    train_spec, eval_spec, _ = self._train_eval_specs()
    tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)
