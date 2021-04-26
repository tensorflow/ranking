description: Class to set up the input, train and eval processes for a TF
Ranking model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.ext.pipeline.RankingPipeline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="train_and_eval"/>
</div>

# tfr.ext.pipeline.RankingPipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/pipeline.py#L20-L408">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Class to set up the input, train and eval processes for a TF Ranking model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.ext.pipeline.RankingPipeline(
    context_feature_columns, example_feature_columns, hparams, estimator,
    label_feature_name=&#x27;relevance&#x27;, label_feature_type=tf.int64,
    dataset_reader=tfr.keras.pipeline.DatasetHparams.dataset_reader,
    best_exporter_metric=None, best_exporter_metric_higher_better=True,
    size_feature_name=None
)
</code></pre>

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

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`context_feature_columns`
</td>
<td>
(dict) Context (aka, query) feature columns.
</td>
</tr><tr>
<td>
`example_feature_columns`
</td>
<td>
(dict) Example (aka, document) feature columns.
</td>
</tr><tr>
<td>
`hparams`
</td>
<td>
(dict) A dict containing model hyperparameters.
</td>
</tr><tr>
<td>
`estimator`
</td>
<td>
(`Estimator`) An `Estimator` instance for model train and eval.
</td>
</tr><tr>
<td>
`label_feature_name`
</td>
<td>
(str) The name of the label feature.
</td>
</tr><tr>
<td>
`label_feature_type`
</td>
<td>
(`tf.dtype`) The value type of the label feature.
</td>
</tr><tr>
<td>
`dataset_reader`
</td>
<td>
(`tf.Dataset`) The dataset format for the input files.
</td>
</tr><tr>
<td>
`best_exporter_metric`
</td>
<td>
(str) Metric key for exporting the best model. If
None, exports the model with the minimal loss value.
</td>
</tr><tr>
<td>
`best_exporter_metric_higher_better`
</td>
<td>
(bool) If a higher metric is better.
This is only used if `best_exporter_metric` is not None.
</td>
</tr><tr>
<td>
`size_feature_name`
</td>
<td>
(str) If set, populates the feature dictionary with
this name and the coresponding value is a `tf.int32` Tensor of shape
[batch_size] indicating the actual sizes of the example lists before
padding and truncation. If None, which is default, this feature is not
generated.
</td>
</tr>
</table>

## Methods

<h3 id="train_and_eval"><code>train_and_eval</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/pipeline.py#L401-L408">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train_and_eval(
    local_training=True
)
</code></pre>

Launches train and evaluation jobs locally.
