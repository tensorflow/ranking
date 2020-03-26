<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.ext.pipeline.RankingPipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="train_and_eval"/>
</div>

# tfr.ext.pipeline.RankingPipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/pipeline.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Class to set up the input, train and eval processes for a TF Ranking model.

```python
tfr.ext.pipeline.RankingPipeline(
    context_feature_columns, example_feature_columns, hparams, estimator,
    label_feature_name='relevance', label_feature_type=tf.int64,
    dataset_reader=tf.data.TFRecordDataset, best_exporter_metric=None,
    best_exporter_metric_higher_better=True, size_feature_name=None
)
```

<!-- Placeholder for "Used in" -->

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

#### Note that you may:

*   pass `best_exporter_metric` and `best_exporter_metric_higher_better` for
    different model export strategies.
*   pass `dataset_reader` for reading different `tf.Dataset`s. We recommend
    using TFRecord files and storing your data in
    <a href="../../../tfr/data.md#ELWC"><code>tfr.data.ELWC</code></a> format.

If you want to further customize certain `RankingPipeline` behaviors, please
create a subclass of `RankingPipeline`, and overwrite related functions. We
recommend only overwriting the following functions: * `_make_dataset` which
builds the tf.dataset for a tf-ranking model. * `_make_serving_input_fn` that
defines the input function for serving. * `_export_strategies` if you have more
advanced needs for model exporting.

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

#### Args:

*   <b>`context_feature_columns`</b>: (dict) Context (aka, query) feature
    columns.
*   <b>`example_feature_columns`</b>: (dict) Example (aka, document) feature
    columns.
*   <b>`hparams`</b>: (dict) A dict containing model hyperparameters.
*   <b>`estimator`</b>: (`Estimator`) An `Estimator` instance for model train
    and eval.
*   <b>`label_feature_name`</b>: (str) The name of the label feature.
*   <b>`label_feature_type`</b>: (`tf.dtype`) The value type of the label
    feature.
*   <b>`dataset_reader`</b>: (`tf.Dataset`) The dataset format for the input
    files.
*   <b>`best_exporter_metric`</b>: (str) Metric key for exporting the best
    model. If None, exports the model with the minimal loss value.
*   <b>`best_exporter_metric_higher_better`</b>: (bool) If a higher metric is
    better. This is only used if `best_exporter_metric` is not None.
*   <b>`size_feature_name`</b>: (str) If set, populates the feature dictionary
    with this name and the coresponding value is a `tf.int32` Tensor of shape
    [batch_size] indicating the actual sizes of the example lists before padding
    and truncation. If None, which is default, this feature is not generated.

## Methods

<h3 id="train_and_eval"><code>train_and_eval</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/pipeline.py">View
source</a>

```python
train_and_eval(
    local_training=True
)
```

Launches train and evaluation jobs locally.
