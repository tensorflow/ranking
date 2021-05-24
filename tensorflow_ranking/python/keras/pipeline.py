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
"""Ranking pipeline to train tf.keras.Model in tfr.keras."""

import abc
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import dataclasses
import tensorflow as tf

from tensorflow_ranking.python import data
from tensorflow_ranking.python.keras import losses
from tensorflow_ranking.python.keras import metrics
from tensorflow_ranking.python.keras import model as model_lib
from tensorflow_ranking.python.keras import saved_model
from tensorflow_ranking.python.keras import strategy_utils


class AbstractPipeline(metaclass=abc.ABCMeta):
  """Interface for ranking pipeline to train a `tf.keras.Model`.

  The `AbstractPipeline` class is an abstract class to train and validate a
  ranking model in tfr.keras.

  To be implemented by subclasses:

    * `build_loss()`: Contains the logic to build a `tf.keras.losses.Loss` or a
    dict or list of `tf.keras.losses.Loss`s to be optimized in training.
    * `build_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s to monitor and evaluate the training.
    * `build_weighted_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s which will take the weights.
    * `train_and_validate()`: Contrains the main training pipeline for training
    and validation.

  Example subclass implementation:

  ```python
  class BasicPipeline(AbstractPipeline):

    def __init__(self, model, train_data, valid_data, name=None):
      self._model = model
      self._train_data = train_data
      self._valid_data = valid_data
      self._name = name

    def build_loss(self):
      return tfr.keras.losses.get('softmax_loss')

    def build_metrics(self):
      return [
          tfr.keras.metrics.get(
              'ndcg', topn=topn, name='ndcg_{}'.format(topn)
          ) for topn in [1, 5, 10]
      ]

    def build_weighted_metrics(self):
      return [
          tfr.keras.metrics.get(
              'ndcg', topn=topn, name='weighted_ndcg_{}'.format(topn)
          ) for topn in [1, 5, 10]
      ]

    def train_and_validate(self, *arg, **kwargs):
      self._model.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
          loss=self.build_loss(),
          metrics=self.build_metrics(),
          weighted_metrics=self.build_weighted_metrics())
      self._model.fit(
          x=self._train_data,
          epochs=100,
          validation_data=self._valid_data)
  ```
  """

  @abc.abstractmethod
  def build_loss(self) -> Any:
    """Returns the loss for model.compile.

    Example usage:

    ```python
    pipeline = BasicPipeline(model, train_data, valid_data)
    loss = pipeline.build_loss()
    ```

    Returns:
      A `tf.keras.losses.Loss` or a dict or list of `tf.keras.losses.Loss`.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def build_metrics(self) -> Any:
    """Returns a list of ranking metrics for `model.compile()`.

    Example usage:

    ```python
    pipeline = BasicPipeline(model, train_data, valid_data)
    metrics = pipeline.build_metrics()
    ```

    Returns:
      A list or a dict of `tf.keras.metrics.Metric`s.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def build_weighted_metrics(self) -> Any:
    """Returns a list of weighted ranking metrics for model.compile.

    Example usage:

    ```python
    pipeline = BasicPipeline(model, train_data, valid_data)
    weighted_metrics = pipeline.build_weighted_metrics()
    ```

    Returns:
      A list or a dict of `tf.keras.metrics.Metric`s.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def train_and_validate(self, *arg, **kwargs) -> Any:
    """Constructs and runs the training pipeline.

    Example usage:

    ```python
    pipeline = BasicPipeline(model, train_data, valid_data)
    pipeline.train_and_validate()
    ```

    Args:
      *arg: arguments that might be used in the training pipeline.
      **kwargs: keyword arguments that might be used in the training pipeline.

    Returns:
      None or a trained `tf.keras.Model` or a path to a saved `tf.keras.Model`.
    """
    raise NotImplementedError("Calling an abstract method.")


class AbstractDatasetBuilder(metaclass=abc.ABCMeta):
  """Interface for datasets and signatures.

  The `AbstractDatasetBuilder` class is an abstract class to serve data in
  tfr.keras. A `DatasetBuilder` will be passed to an instance of
  `AbstractPipeline` and called to serve the training and validation datasets
  and to define the serving signatures for saved models to treat the
  corresponding format of data.

  To be implemented by subclasses:

    * `build_train_dataset()`: Contains the logic to build a `tf.data.Dataset`
    for training.
    * `build_valid_dataset()`: Contains the logic to build a `tf.data.Dataset`
    for validation.
    * `build_signatures()`: Contains the logic to build a dict of signatures
    that formulate the model in functions that render the input data with given
    format.

  Example subclass implementation:

  ```python
  class NullDatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, train_dataset, valid_dataset, signatures=None):
      self._train_dataset = train_dataset
      self._valid_dataset = valid_dataset
      self._signatures = signatures

    def build_train_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
      return self._train_dataset

    def build_valid_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
      return self._valid_dataset

    def build_signatures(self, *arg, **kwargs) -> Any:
      return self._signatures
  ```
  """

  @abc.abstractmethod
  def build_train_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
    """Returns the training dataset.

    Example usage:

    ```python
    dataset_builder = NullDatasetBuilder(train_data, valid_data)
    train_dataset = dataset_builder.build_train_dataset()
    ```

    Args:
      *arg: arguments that might be used to build training dataset.
      **kwargs: keyword arguments that might be used to build training dataset.

    Returns:
      A `tf.data.Dataset`.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def build_valid_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
    """Returns the validation dataset.

    Example usage:

    ```python
    dataset_builder = NullDatasetBuilder(train_data, valid_data)
    valid_dataset = dataset_builder.build_valid_dataset()
    ```

    Args:
      *arg: arguments that might be used to build validation dataset.
      **kwargs: keyword arguments that might be used to build validation
        dataset.

    Returns:
      A `tf.data.Dataset`.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def build_signatures(self, *arg, **kwargs) -> Any:
    """Returns the signatures to export a SavedModel.

    Example usage:

    ```python
    dataset_builder = NullDatasetBuilder(train_data, valid_data)
    signatures = dataset_builder.build_signatures()
    ```

    Args:
      *arg: arguments that might be used to build signatures.
      **kwargs: keyword arguments that might be used to build signatures.

    Returns:
      None or a dict of concrete functions.
    """
    raise NotImplementedError("Calling an abstract method.")


@dataclasses.dataclass
class PipelineHparams:
  """Hyperparameters used in `ModelFitPipeline`.

  Hyperparameters to be specified for ranking pipeline.

  Attributes:
    model_dir: A path to output the model and training data.
    num_epochs: An integer to specify the number of epochs of training.
    steps_per_epoch: An integer to specify the number of steps per epoch. When
      it is None, going over the training data once is counted as an epoch.
    validation_steps: An integer to specify the number of validation steps in
      each epoch. Note that a mini-batch of data will be evaluated in each step
      and this is the number of steps taken for validation in each epoch.
    learning_rate: A float to indicate the learning rate of the optimizer.
    loss: A string or a map to strings that indicate the loss to be used. When
      `loss` is a string, all outputs and labels will be trained with the same
      loss. When `loss` is a map, outputs and labels will be trained with losses
      implied by the corresponding keys.
    loss_reduction: An option in `tf.keras.losses.Reduction` to specify the
      reduction method.
    optimizer: An option in `tf.keras.optimizers` identifiers to specify the
      optimizer to be used.
    loss_weights: None or a float or a map to floats that indicate the relative
      weights for each loss. When not specified, all losses are applied with the
      same weight 1.
    steps_per_execution: An integer to specify the number of steps executed in
      each operation. Tuning this to optimize the training performance in
      distributed training.
    automatic_reduce_lr: A boolean to indicate whether to use
      `ReduceLROnPlateau` callback.
    use_weighted_metrics: A boolean to indicate whether to use weighted metrics.
    export_best_model: A boolean to indicate whether to export the best model
      evaluated by the `best_exporter_metric` on the validation data.
    best_exporter_metric_higher_better: A boolean to indicate whether the
      `best_exporter_metric` is the higher the better.
    best_exporter_metric: A string to specify the metric used to monitor the
      training and to export the best model. Default to the 'loss'.
    strategy: An option of strategies supported in `strategy_utils`. Choose from
      ["MirroredStrategy", "MultiWorkerMirroredStrategy", "TPUStrategy"].
    tpu: TPU address for TPUStrategy. Not used for other strategy.
  """
  model_dir: str
  num_epochs: int
  steps_per_epoch: int
  validation_steps: int
  learning_rate: float
  loss: Union[str, Dict[str, str]]
  loss_reduction: str = tf.losses.Reduction.AUTO
  optimizer: str = "adam"
  loss_weights: Optional[Union[float, Dict[str, float]]] = None
  steps_per_execution: int = 10
  automatic_reduce_lr: bool = False
  use_weighted_metrics: bool = False
  export_best_model: bool = False
  best_exporter_metric_higher_better: bool = False
  best_exporter_metric: str = "loss"
  strategy: Optional[str] = None
  tpu: Optional[str] = ""


@dataclasses.dataclass
class DatasetHparams:
  """Hyperparameters used in `BaseDatasetBuilder`.

  Hyperparameters to be specified to create the dataset_builder.

  Attributes:
    train_input_pattern: A glob pattern to specify the paths to the input data
      for training.
    valid_input_pattern: A glob pattern to specify the paths to the input data
      for validation.
    train_batch_size: An integer to specify the batch size of training dataset.
    valid_batch_size: An integer to specify the batch size of valid dataset.
    list_size: An integer to specify the list size. When None, data will be
      padded to the longest list in each batch.
    valid_list_size: An integer to specify the list size in valid dataset. When
      not specified, valid dataset uses the same list size as `list_size`.
    dataset_reader: A function or class that can be called with a `filenames`
      tensor and (optional) `reader_args` and returns a `Dataset`. Defaults to
      `tf.data.TFRecordDataset`.
    convert_labels_to_binary: A boolean to indicate whether to use binary label.
  """
  train_input_pattern: str
  valid_input_pattern: str
  train_batch_size: int
  valid_batch_size: int
  list_size: Optional[int] = None
  valid_list_size: Optional[int] = None
  dataset_reader: Any = tf.data.TFRecordDataset
  convert_labels_to_binary: bool = False


class ModelFitPipeline(AbstractPipeline):
  """Pipeline using `model.fit` to train a ranking `tf.keras.Model`.

  The `ModelFitPipeline` class is an abstract class inherit from
  `AbstractPipeline` to train and validate a ranking `model` with `model.fit`
  in a distributed strategy specified in hparams.

  To be implemented by subclasses:

    * `build_loss()`: Contains the logic to build a `tf.keras.losses.Loss` or a
    dict or list of `tf.keras.losses.Loss`s to be optimized in training.
    * `build_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s to monitor and evaluate the training.
    * `build_weighted_metrics()`: Contains the logic to build a list or dict of
    `tf.keras.metrics.Metric`s which will take the weights.

  Example subclass implementation:

  ```python
  class BasicModelFitPipeline(ModelFitPipeline):

    def build_loss(self):
      return tfr.keras.losses.get('softmax_loss')

    def build_metrics(self):
      return [
          tfr.keras.metrics.get(
              'ndcg', topn=topn, name='ndcg_{}'.format(topn)
          ) for topn in [1, 5, 10]
      ]

    def build_weighted_metrics(self):
      return [
          tfr.keras.metrics.get(
              'ndcg', topn=topn, name='weighted_ndcg_{}'.format(topn)
          ) for topn in [1, 5, 10]
      ]
  ```
  """

  def __init__(
      self,
      model_builder: model_lib.AbstractModelBuilder,
      dataset_builder: AbstractDatasetBuilder,
      hparams: PipelineHparams,
  ):
    """Initializes the instance.

    Args:
      model_builder: A `ModelBuilder` instance for model fit.
      dataset_builder: An `AbstractDatasetBuilder` instance to load train and
        validate datasets and create signatures for SavedModel.
      hparams: A dict containing model hyperparameters.
    """
    self._validate_parameters(model_builder, dataset_builder)

    self._model_builder = model_builder
    self._dataset_builder = dataset_builder
    self._hparams = hparams

    self._optimizer = tf.keras.optimizers.get({
        "class_name": self._hparams.optimizer,
        "config": {
            "learning_rate": self._hparams.learning_rate
        }
    })

    self._strategy = strategy_utils.get_strategy(self._hparams.strategy,
                                                 self._hparams.tpu)

  def _validate_parameters(self, model_builder: model_lib.AbstractModelBuilder,
                           dataset_builder: AbstractDatasetBuilder):
    """Validates the passed-in model_builder and dataset_builder.

    Args:
      model_builder: A `ModelBuilder` instance.
      dataset_builder: A `DatasetBuilder` instance.

    Raises:
      ValueError: If the `model_builder` is None.
      ValueError: If the `model_builder` is not an `ModelBuilder`.
      ValueError: If the `dataset_builder` is None.
      ValueError: If the `dataset_builder` is not an `DatasetBuilder`.
    """
    if model_builder is None:
      raise ValueError("The `model_builder` cannot be empty!")

    if not isinstance(model_builder, model_lib.AbstractModelBuilder):
      raise ValueError(
          "The argument `model_builder` needs to be of type "
          "tensorflow_ranking.keras.model.AbstractModelBuilder, not {}.".format(
              type(model_builder)))

    if dataset_builder is None:
      raise ValueError("The `dataset_builder` cannot be empty!")

    if not isinstance(dataset_builder, AbstractDatasetBuilder):
      raise ValueError(
          "The argument `dataset_builder` needs to be of type "
          "tensorflow_ranking.keras.pipeline.DatasetBuilder, not {}.".format(
              type(dataset_builder)))

  def build_callbacks(self) -> List[tf.keras.callbacks.Callback]:
    """Sets up Callbacks.

    Example usage:

    ```python
    model_builder = ModelBuilder(...)
    dataset_builder = DatasetBuilder(...)
    hparams = PipelineHparams(...)
    pipeline = BasicModelFitPipeline(model_builder, dataset_builder, hparams)
    callbacks = pipeline.build_callbacks()
    ```

    Returns:
      A list of `tf.keras.callbacks.Callback` or a
      `tf.keras.callbacks.CallbackList` for tensorboard and checkpoint.
    """
    # Writing summary logs to file may have performance impact. Therefore, we
    # only write summary events every epoch.
    callbacks = [
        tf.keras.callbacks.TensorBoard(self._hparams.model_dir),
        tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=self._hparams.model_dir)
    ]

    if self._hparams.export_best_model:
      # default to be min of loss metric.
      best_export_metric = self._hparams.best_exporter_metric
      if best_export_metric != "loss":
        best_export_metric = "metric/" + best_export_metric
      callbacks.append(
          tf.keras.callbacks.ModelCheckpoint(
              os.path.join(self._hparams.model_dir,
                           "best_checkpoint/ckpt-{epoch:04d}"),
              monitor="val_" + best_export_metric,
              mode=("max" if self._hparams.best_exporter_metric_higher_better
                    else "min"),
              save_weights_only=True,
              save_best_only=True))

    if self._hparams.automatic_reduce_lr:
      callbacks.append(
          tf.keras.callbacks.ReduceLROnPlateau(
              monitor="val_loss",
              min_delta=0.01 * self._hparams.learning_rate,
          ))

    return callbacks

  def export_saved_model(self,
                         model: tf.keras.Model,
                         export_to: str,
                         checkpoint: Optional[tf.train.Checkpoint] = None):
    """Exports the trained model with signatures.

    Example usage:

    ```python
    model_builder = ModelBuilder(...)
    dataset_builder = DatasetBuilder(...)
    hparams = PipelineHparams(...)
    pipeline = BasicModelFitPipeline(model_builder, dataset_builder, hparams)
    pipeline.export_saved_model(model_builder.build(), 'saved_model/')
    ```

    Args:
      model: Model to be saved.
      export_to: Specifies the directory the model is be exported to.
      checkpoint: If given, export the model with weights from this checkpoint.
    """
    if checkpoint:
      model.load_weights(checkpoint)
    model.save(
        filepath=export_to,
        signatures=self._dataset_builder.build_signatures(model))

  def train_and_validate(self, verbose=0):
    """Main function to train the model with TPU strategy.

    Example usage:

    ```python
    context_feature_spec = {}
    example_feature_spec = {
        "example_feature_1": tf.io.FixedLenFeature(
            shape=(1,), dtype=tf.float32, default_value=0.0)
    }
    mask_feature_name = "list_mask"
    label_spec = {
        "utility": tf.io.FixedLenFeature(
            shape=(1,), dtype=tf.float32, default_value=0.0)
    }
    dataset_hparams = DatasetHparams(
        train_input_pattern="train.dat",
        valid_input_pattern="valid.dat",
        train_batch_size=128,
        valid_batch_size=128)
    pipeline_hparams = pipeline.PipelineHparams(
        model_dir="model/",
        num_epochs=2,
        steps_per_epoch=5,
        validation_steps=2,
        learning_rate=0.01,
        loss="softmax_loss")
    model_builder = SimpleModelBuilder(
        context_feature_spec, example_feature_spec, mask_feature_name)
    dataset_builder = SimpleDatasetBuilder(
        context_feature_spec,
        example_feature_spec,
        mask_feature_name,
        label_spec,
        dataset_hparams)
    pipeline = BasicModelFitPipeline(
        model_builder, dataset_builder, pipeline_hparams)
    pipeline.train_and_validate(verbose=1)
    ```

    Args:
      verbose: An int for the verbosity level.
    """
    strategy = self._strategy
    with strategy_utils.strategy_scope(strategy):
      model = self._model_builder.build()
      # Note that all losses and metrics need to be constructed within the
      # strategy scope. This is why we use member function like `build_loss` and
      # don't use passed-in objects.
      model.compile(
          optimizer=self._optimizer,
          loss=self.build_loss(),
          metrics=self.build_metrics(),
          loss_weights=self._hparams.loss_weights,
          weighted_metrics=(self.build_weighted_metrics()
                            if self._hparams.use_weighted_metrics else None),
          steps_per_execution=self._hparams.steps_per_execution)

      # Move the following out of strategy.scope only after b/173547275 fixed.
      # Otherwise, MultiWorkerMirroredStrategy will fail.
      train_dataset, valid_dataset = (
          self._dataset_builder.build_train_dataset(),
          self._dataset_builder.build_valid_dataset())
      model.fit(
          x=train_dataset,
          epochs=self._hparams.num_epochs,
          steps_per_epoch=self._hparams.steps_per_epoch,
          validation_steps=self._hparams.validation_steps,
          validation_data=valid_dataset,
          callbacks=self.build_callbacks(),
          verbose=verbose)

      model_output_dir = strategy_utils.get_output_filepath(
          self._hparams.model_dir, strategy)
      self.export_saved_model(
          model,
          export_to=os.path.join(model_output_dir, "export/latest_model"))

      if self._hparams.export_best_model:
        best_checkpoint = tf.train.latest_checkpoint(
            os.path.join(self._hparams.model_dir, "best_checkpoint"))
        if best_checkpoint:
          self.export_saved_model(
              model,
              export_to=os.path.join(model_output_dir,
                                     "export/best_model_by_metric"),
              checkpoint=best_checkpoint)
        else:
          raise ValueError("Didn't find the best checkpoint.")


def _get_metric(prefix, key, topn=None):
  """Helper function to construct a metric."""
  name = "{}{}{}".format(prefix, key, "_%s" % topn if topn else "")
  return metrics.get(key, name=name, topn=topn)


class SimplePipeline(ModelFitPipeline):
  """Pipleine for single-task training.

  This handles a single loss and works with `SimpleDatasetBuilder`. This can
  also work with `MultiLabelDatasetBuilder`. In this case, the same loss, as
  well as all metrics, will be applied to all labels and their predictions
  uniformly.

  Use subclassing to customize the loss and metrics.

  Example usage:

  ```python
  context_feature_spec = {}
  example_feature_spec = {
      "example_feature_1": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)
  }
  mask_feature_name = "list_mask"
  label_spec = {
      "utility": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)
  }
  dataset_hparams = DatasetHparams(
      train_input_pattern="train.dat",
      valid_input_pattern="valid.dat",
      train_batch_size=128,
      valid_batch_size=128)
  pipeline_hparams = pipeline.PipelineHparams(
      model_dir="model/",
      num_epochs=2,
      steps_per_epoch=5,
      validation_steps=2,
      learning_rate=0.01,
      loss="softmax_loss")
  model_builder = SimpleModelBuilder(
      context_feature_spec, example_feature_spec, mask_feature_name)
  dataset_builder = SimpleDatasetBuilder(
      context_feature_spec,
      example_feature_spec,
      mask_feature_name,
      label_spec,
      dataset_hparams)
  pipeline = SimplePipeline(model_builder, dataset_builder, pipeline_hparams)
  pipeline.train_and_validate(verbose=1)
  ```
  """

  def build_loss(self) -> tf.keras.losses.Loss:
    """See `AbstractPipeline`."""
    if isinstance(self._hparams.loss, dict):
      raise TypeError("In the simple pipeline, losses are expected to be "
                      "specified in a str instead of a dict.")
    return losses.get(
        loss=self._hparams.loss, reduction=self._hparams.loss_reduction)

  def build_metrics(self) -> List[tf.keras.metrics.Metric]:
    """See `AbstractPipeline`."""
    eval_metrics = [
        _get_metric("metric/", metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 5, 10, None]
    ]
    return eval_metrics

  def build_weighted_metrics(self) -> List[tf.keras.metrics.Metric]:
    """See `AbstractPipeline`."""
    eval_metrics = [
        _get_metric(
            "weighted_metric/", metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 5, 10, None]
    ]
    return eval_metrics


class MultiTaskPipeline(ModelFitPipeline):
  """Pipeline for multi-task training.

  This handles a set of losses and labels. It is intended to mainly work with
  `MultiLabelDatasetBuilder`.

  Use subclassing to customize the losses and metrics.

  Example usage:

  ```python
  context_feature_spec = {}
  example_feature_spec = {
      "example_feature_1": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)
  }
  mask_feature_name = "list_mask"
  label_spec_tuple = ("utility",
                      tf.io.FixedLenFeature(
                          shape=(1,),
                          dtype=tf.float32,
                          default_value=_PADDING_LABEL))
  label_spec = {"task1": label_spec_tuple, "task2": label_spec_tuple}
  weight_spec = ("weight",
                 tf.io.FixedLenFeature(
                     shape=(1,), dtype=tf.float32, default_value=1.))
  dataset_hparams = DatasetHparams(
      train_input_pattern="train.dat",
      valid_input_pattern="valid.dat",
      train_batch_size=128,
      valid_batch_size=128)
  pipeline_hparams = PipelineHparams(
      model_dir="model/",
      num_epochs=2,
      steps_per_epoch=5,
      validation_steps=2,
      learning_rate=0.01,
      loss={
          "task1": "softmax_loss",
          "task2": "pairwise_logistic_loss"
      },
      loss_weights={
          "task1": 1.0,
          "task2": 2.0
      },
      export_best_model=True)
  model_builder = MultiTaskModelBuilder(...)
  dataset_builder = MultiLabelDatasetBuilder(
      context_feature_spec,
      example_feature_spec,
      mask_feature_name,
      label_spec,
      dataset_hparams,
      sample_weight_spec=weight_spec)
  pipeline = MultiTaskPipeline(model_builder, dataset_builder, pipeline_hparams)
  pipeline.train_and_validate(verbose=1)
  ```
  """

  def build_loss(self) -> Dict[str, tf.keras.losses.Loss]:
    """See `AbstractPipeline`."""
    reduction = self._hparams.loss_reduction
    if isinstance(self._hparams.loss, str):
      raise TypeError("In the multi-task pipeline, losses are expected to be "
                      "specified in a dict instead of a str.")
    return {
        task_name: losses.get(loss=loss_name, reduction=reduction)
        for task_name, loss_name in self._hparams.loss.items()
    }

  def build_metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
    """See `AbstractPipeline`."""

    def eval_metrics():
      return [
          _get_metric("metric/", metrics.RankingMetricKey.NDCG, topn=topn)
          for topn in [1, 5, 10, None]
      ]

    return {task_name: eval_metrics() for task_name in self._hparams.loss}

  def build_weighted_metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
    """See `AbstractPipeline`."""

    def eval_metrics():
      return [
          _get_metric(
              "weighted_metric/", metrics.RankingMetricKey.NDCG, topn=topn)
          for topn in [1, 5, 10, None]
      ]

    return {task_name: eval_metrics() for task_name in self._hparams.loss}


class NullDatasetBuilder(AbstractDatasetBuilder):
  """A no-op wrapper of datasets and signatures.

  Example usage:

  ```python
  train_dataset = tf.data.Dataset(...)
  valid_dataset = tf.data.Dataset(...)
  dataset_builder = NullDatasetBuilder(train_dataset, valid_dataset)
  ```
  """

  def __init__(self, train_dataset, valid_dataset, signatures=None):
    """Initializes the instance.

    Args:
      train_dataset: A `tf.data.Dataset` for training.
      valid_dataset: A `tf.data.Dataset` for validation.
      signatures: A dict of signatures that formulate the model in functions
        that render the input data with given types. When None, no signatures
        assigned.
    """
    self._train_dataset = train_dataset
    self._valid_dataset = valid_dataset
    self._signatures = signatures

  def build_train_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
    """See `AbstractDatasetBuilder`."""
    return self._train_dataset

  def build_valid_dataset(self, *arg, **kwargs) -> tf.data.Dataset:
    """See `AbstractDatasetBuilder`."""
    return self._valid_dataset

  def build_signatures(self, *arg, **kwargs) -> Any:
    """See `AbstractDatasetBuilder`."""
    return self._signatures


class BaseDatasetBuilder(AbstractDatasetBuilder):
  """Builds datasets from feature specs.

  The `BaseDatasetBuilder` class is an abstract class inherit from
  `AbstractDatasetBuilder` to serve training and validation datasets and
  signatures for training `ModelFitPipeline`.

  To be implemented by subclasses:

    * `_features_and_labels()`: Contains the logic to map a dict of tensors of
    dataset to feature tensors and label tensors.

  Example subclass implementation:

  ```python
  class SimpleDatasetBuilder(BaseDatasetBuilder):

    def _features_and_labels(self, features):
      label = features.pop("utility")
      return features, label
  ```
  """

  # TODO: Define these bulky types as globals at the top.
  def __init__(self,
               context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                                     tf.io.VarLenFeature,
                                                     tf.io.RaggedFeature]],
               example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                                     tf.io.VarLenFeature,
                                                     tf.io.RaggedFeature]],
               training_only_example_spec: Dict[str,
                                                Union[tf.io.FixedLenFeature,
                                                      tf.io.VarLenFeature,
                                                      tf.io.RaggedFeature]],
               mask_feature_name: str,
               hparams: DatasetHparams,
               training_only_context_spec: Optional[Dict[
                   str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature,
                              tf.io.RaggedFeature]]] = None):
    """Intializes the instance.

    Args:
      context_feature_spec: Maps context (aka, query) names to feature specs.
      example_feature_spec: Maps example (aka, document) names to feature specs.
      training_only_example_spec: Feature specs used for training only like
        labels and per-example weights.
      mask_feature_name: If set, populates the feature dictionary with this name
        and the coresponding value is a `tf.bool` Tensor of shape [batch_size,
        list_size] indicating the actual example is padded or not.
      hparams: A dict containing model hyperparameters.
      training_only_context_spec: Feature specs used for training only per-list
        weights.
    """
    self._context_feature_spec = context_feature_spec
    self._example_feature_spec = example_feature_spec
    self._training_only_example_spec = training_only_example_spec
    self._mask_feature_name = mask_feature_name
    self._hparams = hparams
    self._training_only_context_spec = training_only_context_spec or {}

  @abc.abstractmethod
  def _features_and_labels(self, features: Dict[str, tf.Tensor]) -> Any:
    """Extracts labels and weights from features.

    Args:
      features: Maps feature name and label name to corresponding tensors.

    Returns:
      A tuple of a dict of the rest of features, labels and optional weights.
    """
    raise NotImplementedError("Calling an abstract method.")

  def _build_dataset(self,
                     file_pattern: str,
                     batch_size: int,
                     list_size: Optional[int] = None,
                     randomize_input: bool = True,
                     num_epochs: Optional[int] = None) -> tf.data.Dataset:
    """Returns `tf.data.Dataset` for training or validating the model.

    Args:
      file_pattern: File pattern for input data.
      batch_size: Number of input examples to process per batch.
      list_size: The list size for an ELWC example.
      randomize_input: If true, randomize input example order. It should almost
        always be true except for unittest/debug purposes.
      num_epochs: Number of times the input dataset must be repeated. None to
        repeat the data indefinitely.

    Returns:
      A `tf.data.Dataset`.
    """
    # TODO: Remove defaults common in Estimator pipeline and here.
    dataset = data.build_ranking_dataset(
        file_pattern=file_pattern,
        data_format=data.ELWC,
        batch_size=batch_size,
        list_size=list_size,
        context_feature_spec=dict(
            list(self._context_feature_spec.items()) +
            list(self._training_only_context_spec.items())),
        example_feature_spec=dict(
            list(self._example_feature_spec.items()) +
            list(self._training_only_example_spec.items())),
        mask_feature_name=self._mask_feature_name,
        reader=self._hparams.dataset_reader,
        reader_args=None,
        num_epochs=num_epochs,
        shuffle=randomize_input,
        shuffle_buffer_size=1000,
        shuffle_seed=None,
        prefetch_buffer_size=10000,
        reader_num_threads=64,
        sloppy_ordering=True,
        drop_final_batch=False,
        shuffle_examples=False)

    return dataset.map(
        self._features_and_labels,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def build_train_dataset(self) -> tf.data.Dataset:
    """See `AbstractDatasetBuilder`."""
    train_list_size = self._hparams.list_size
    return self._build_dataset(
        file_pattern=self._hparams.train_input_pattern,
        batch_size=self._hparams.train_batch_size,
        list_size=train_list_size)

  def build_valid_dataset(self) -> tf.data.Dataset:
    """See `AbstractDatasetBuilder`."""
    valid_list_size = (self._hparams.valid_list_size or self._hparams.list_size)
    return self._build_dataset(
        file_pattern=self._hparams.valid_input_pattern,
        batch_size=self._hparams.valid_batch_size,
        list_size=valid_list_size,
        randomize_input=False)

  def build_signatures(self, model: tf.keras.Model) -> Any:
    """See `AbstractDatasetBuilder`."""
    return saved_model.Signatures(
        model,
        context_feature_spec=self._context_feature_spec,
        example_feature_spec=self._example_feature_spec,
        mask_feature_name=self._mask_feature_name)()


def _convert_label(label, convert_labels_to_binary=False):
  """Converts the label to 2D and optionally binarizes it.

  Args:
    label: A tensor of shape [batch_size, list_size, 1].
    convert_labels_to_binary: A boolean to indicate whether to use binary label.

  Returns:
    A tensor of shape [batch_size, list_size].
  """
  label = tf.cast(tf.squeeze(label, axis=2), tf.float32)
  if convert_labels_to_binary:
    label = tf.where(tf.greater(label, 0.), tf.ones_like(label), label)
  return label


class SimpleDatasetBuilder(BaseDatasetBuilder):
  """Builds datasets from feature specs with a single label spec.

  This supports a single dataset with a single label, which is supposed to be a
  dense Tensor.

  Example usage:

  ```python
  context_feature_spec = {}
  example_feature_spec = {
      "example_feature_1": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)
  }
  mask_feature_name = "list_mask"
  label_spec = {
      "utility": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)
  }
  dataset_hparams = DatasetHparams(
      train_input_pattern="train.dat",
      valid_input_pattern="valid.dat",
      train_batch_size=128,
      valid_batch_size=128)
  dataset_builder = SimpleDatasetBuilder(
      context_feature_spec,
      example_feature_spec,
      mask_feature_name,
      label_spec,
      dataset_hparams)
  ```
  """

  def __init__(
      self,
      context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
      example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
      mask_feature_name: str,
      label_spec: Tuple[str, tf.io.FixedLenFeature],
      hparams: DatasetHparams,
      sample_weight_spec: Optional[Tuple[str, tf.io.FixedLenFeature]] = None):
    """Intializes the instance.

    Args:
      context_feature_spec: Maps context (aka, query) names to feature specs.
      example_feature_spec: Maps example (aka, document) names to feature specs.
      mask_feature_name: If set, populates the feature dictionary with this name
        and the coresponding value is a `tf.bool` Tensor of shape [batch_size,
        list_size] indicating the actual example is padded or not.
      label_spec: A tuple of the label name and a tf.io.FixedLenFeature spec, or
        a dict that maps task name to label spec in multi-task setting.
      hparams: A dict containing model hyperparameters.
      sample_weight_spec: Feature spec for per-example weight.
    """
    super().__init__(
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        training_only_example_spec=dict([label_spec] + (
            [sample_weight_spec] if sample_weight_spec else [])),
        mask_feature_name=mask_feature_name,
        hparams=hparams)
    self._label_spec = label_spec
    self._sample_weight_spec = sample_weight_spec

  def _features_and_labels(
      self, features: Dict[str, tf.Tensor]
  ) -> Union[Tuple[Dict[str, tf.Tensor], tf.Tensor],  #
             Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]]:
    """See `BaseDatasetBuilder`."""
    to_pop = set()
    feature_name, _ = self._label_spec
    label = _convert_label(features[feature_name],
                           self._hparams.convert_labels_to_binary)
    to_pop.add(feature_name)
    if self._sample_weight_spec:
      feature_name, _ = self._sample_weight_spec
      weight = tf.cast(tf.squeeze(features[feature_name], 2), tf.float32)
      to_pop.add(feature_name)
    else:
      weight = None

    for name in to_pop:
      features.pop(name)

    if weight is None:
      return features, label
    else:
      return features, label, weight


class MultiLabelDatasetBuilder(BaseDatasetBuilder):
  """Builds datasets for multi-task training.

  This supports a single data sets with multiple labels formed in a dict. The
  case where we have multiple datasets is not handled in the current code yet.
  We can consider to extend the dataset builder when the use case comes out.

  Example usage:

  ```python
  context_feature_spec = {}
  example_feature_spec = {
      "example_feature_1": tf.io.FixedLenFeature(
          shape=(1,), dtype=tf.float32, default_value=0.0)
  }
  mask_feature_name = "list_mask"
  label_spec_tuple = ("utility",
                      tf.io.FixedLenFeature(
                          shape=(1,),
                          dtype=tf.float32,
                          default_value=_PADDING_LABEL))
  label_spec = {"task1": label_spec_tuple, "task2": label_spec_tuple}
  weight_spec = ("weight",
                 tf.io.FixedLenFeature(
                     shape=(1,), dtype=tf.float32, default_value=1.))
  dataset_hparams = DatasetHparams(
      train_input_pattern="train.dat",
      valid_input_pattern="valid.dat",
      train_batch_size=128,
      valid_batch_size=128)
  dataset_builder = MultiLabelDatasetBuilder(
      context_feature_spec,
      example_feature_spec,
      mask_feature_name,
      label_spec,
      dataset_hparams,
      sample_weight_spec=weight_spec)
  ```
  """

  def __init__(
      self,
      context_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
      example_feature_spec: Dict[str, Union[tf.io.FixedLenFeature,
                                            tf.io.VarLenFeature,
                                            tf.io.RaggedFeature]],
      mask_feature_name: str,
      label_spec: Dict[str, Tuple[str, tf.io.FixedLenFeature]],
      hparams: DatasetHparams,
      sample_weight_spec: Optional[Tuple[str, tf.io.FixedLenFeature]] = None):
    """Intializes the instance.

    Args:
      context_feature_spec: Maps context (aka, query) names to feature specs.
      example_feature_spec: Maps example (aka, document) names to feature specs.
      mask_feature_name: If set, populates the feature dictionary with this name
        and the coresponding value is a `tf.bool` Tensor of shape [batch_size,
        list_size] indicating the actual example is padded or not.
      label_spec: A dict that maps task names to label specs. Each of the latter
        have a label name and a tf.io.FixedLenFeature spec.
      hparams: A dict containing model hyperparameters.
      sample_weight_spec: Feature spec for per-example weight.
    """
    super().__init__(
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        training_only_example_spec=dict(
            list(label_spec.values()) +
            ([sample_weight_spec] if sample_weight_spec else [])),
        mask_feature_name=mask_feature_name,
        hparams=hparams)
    self._label_spec = label_spec
    self._sample_weight_spec = sample_weight_spec

  def _features_and_labels(
      self, features: Dict[str, tf.Tensor]
  ) -> Union[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],  #
             Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor]]:
    """See `BaseDatasetBuilder`."""
    to_pop = set()
    label = {}
    for task_name, (feature_name, _) in self._label_spec.items():
      label[task_name] = _convert_label(features[feature_name],
                                        self._hparams.convert_labels_to_binary)
      to_pop.add(feature_name)

    if self._sample_weight_spec:
      feature_name, _ = self._sample_weight_spec
      weight = tf.cast(tf.squeeze(features[feature_name], 2), tf.float32)
      to_pop.add(feature_name)
    else:
      weight = None

    for name in to_pop:
      features.pop(name)

    if weight is None:
      return features, label
    else:
      return features, label, weight
