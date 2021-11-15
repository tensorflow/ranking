# TF-Ranking Extensions
This file describes a set of TF-Ranking extensions.

[TOC]

## `RankingPipeline` Extension
We firstly provide a step-by-step guide on quickly developing a TF-Ranking model
using `RankingPipeline`. Overall, there are three steps as shown in below.

### Setup
To use `RankingPipeline`, you need to first install the Tensorflow Ranking
library with pip ([instruction](https://github.com/tensorflow/ranking#stable-builds)).

### Step 1: Defining hparams
The `RankingPipeline` requires a dictionary of hparams to construct an instance.
Most of them are standard Tensorflow model parameters, including:

*  `train_input_pattern`, `eval_input_pattern`.
*  `train_batch_size`, `eval_batch_size`.
*  `checkpoint_secs`, `num_checkpoints`.
*  `num_train_steps`, `num_eval_steps`.
*  `model_dir`.

In the meantime, it should also involve TF-Ranking specific parameters such as:

*  `loss`. Here is the list of the supported [ranking losses](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/python/losses.py#L31),
*  `list_size`. Default to be `None`. If `None`, the actual list size per batch
   is determined by the maximum number of examples in that batch. If a fixed
   `list_size` is set, the number of examples larger than the `list_size` will
   be trucated, and smaller than the `list_size` will be padded. Please refer to
   `tfr.data.make_parsing_fn` for more details.
*  `listwise_inference`. Default to be `False`. If `False`, the features will be
   encoded using `tf.feature.encode_pointwise_features`; otherwise, the features
   will be encoded by `tf.feature.encode_listwise_features`.

In addition, we add a parameter `convert_labels_to_binary` for label processing.
If the `convert_labels_to_binary` is set to be True, we convert a label to `1`
if its value is larger than 0; otherwise, it will be converted to `0`. With the
`convert_labels_to_binary` setting to be False, we keep label values intact.

```python
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
```

If there are needs to change the default values, you may pass the parameters
through flags. For example, if you need a `list_size` of 100 and the loss type
to be `approx_ndcg_loss`, you can do the following:

```bash
bazel build -c opt \
  tensorflow_ranking/extension/examples/pipeline_example_py_binary && \
./bazel-bin/tensorflow_ranking/extension/examples/pipeline_example_py_binary \
  --train_input_pattern=/path/to/train/files \
  --eval_input_pattern=/path/to/eval/files \
  --model_dir=/path/to/your/model/directory \
  --list_size=100 \
  --loss="approx_ndcg_loss"
```

### Step 2: Building an `Estimator`
`RankingPipeline` requires a `tf.estimator.Estimator` to define TF-Ranking model
computation details. The simplest approach is to construct an `Estimator` using
`tfr.estimator.EstimatorBuilder`. The following provides a simplfied version of
`pipeline_example.py` using the `RankingPipeline`. With a minimal definition of
`context_feature_columns`, `example_feature_columns` and `scoring_function`, one
can create a `tf.estimator.Estimator`.

```python
import tensorflow as tf
import tensorflow_ranking as tfr

def context_feature_columns():
  return {}

def example_feature_columns():
  sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="document_tokens", hash_bucket_size=100)
  document_embedding = tf.feature_column.embedding_column(
      categorical_column=sparse_column, dimension=20)
  return {"document_tokens": document_embedding}

def scoring_function(context_features, example_features, mode):
  """A feed-forward network to score query-document pairs."""
  input_layer = [
      tf.compat.v1.layers.flatten(context_features["document_tokens"])
  ]
  cur_layer = tf.concat(input_layer, 1)
  for dim in [256, 128, 64, 32]:
    cur_layer = tf.compat.v1.layers.dense(cur_layer, units=dim)
    cur_layer = tf.nn.relu(cur_layer)
  return tf.compat.v1.layers.dense(cur_layer, units=1)

ranking_estimator = tfr.estimator.EstimatorBuilder(
      context_feature_columns(),
      example_feature_columns(),
      scoring_function=scoring_function,
      hparams=hparams).make_estimator()
```

One benefit of using `tfr.estimator.EstimatorBuilder` is that it allows clients
to overwirte default model behaviors. E.g, if you need evaluation metrics other
than the default NDCG@5 and NDCG@10. You may overwrite the `eval_metrics_fn` as
below to obtain MAP@10 and NDCG@20. For function overwriting, please refer to
the constructor of `tfr.estimator.EstimatorBuilder` for more details.

```python
class MyEstimatorBuilder(tfr.estimator.EstimatorBuilder):
  def _eval_metric_fns(self):
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg_%d" % topn: metrics.make_ranking_metric_fn(
            metrics.RankingMetricKey.NDCG, topn=topn) for topn in [20]
    })
    metric_fns.update({
        "metric/ndcg_%d" % topn: metrics.make_ranking_metric_fn(
            metrics.RankingMetricKey.MAP, topn=topn) for topn in [10]
    })
    return metric_fns

ranking_estimator = MyEstimatorBuilder(
      context_feature_columns(),
      example_feature_columns(),
      scoring_function=scoring_function,
      hparams=hparams).make_estimator()
```

If `tfr.estimator.EstimatorBuilder` still cannot fulfill your needs. You can
create your own `tf.estimator.Estimator`. Here, you might still want to refer
to the `tfr.estimator.EstimatorBuilder` as it provides a template for building
an `Estimator`.

### Step 3: Creating an `RankingPipeline` instance
With the above `tf.estimator.Estimator`, `hparams`, `context_feature_columns()`
and `example_feature_columns()`, the next step is to create a `RankingPipeline`
instance, and simply call the `train_and_eval()` for model training and eval.

```python
import tensorflow_ranking as tfr

ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
      context_feature_columns(),
      example_feature_columns(),
      hparams=hparams,
      estimator=ranking_estimator,
      label_feature_name="utility",
      label_feature_type=tf.float32)
ranking_pipeline.train_and_eval()
```

Again, if you want to customize certain `RankingPipeline` behaviors, you may
create a subclass of `RankingPipeline`, and overwrite related functions. For
example, if you want to remove the best exporters, you may do the following.
For function overwriting, please refer to the constructor of `RankingPipeline`
for more details.

```python
class NoBestExporterRankingPipeline(tfr.ext.pipeline.RankingPipeline):
  def _export_strategies(self, event_file_pattern):
    del event_file_pattern
    latest_exporter = tf.estimator.LatestExporter(
        "latest_model",
        serving_input_receiver_fn=self._make_serving_input_fn())
    return [latest_exporter]

ranking_pipeline = NoBestExporterRankingPipeline(
      context_feature_columns(),
      example_feature_columns(),
      hparams=hparams,
      estimator=ranking_estimator)
ranking_pipeline.train_and_eval()
```

### Putting it all together
To summarize the above steps, and put everything together. A complete example
code is provided below. Here, we only provide a simplified illustration for
adopting the `RankingPipeline`. For a running example with the provided data,
please refer to the `pipeline_example.py`.

```python
import tensorflow as tf
import tensorflow_ranking as tfr

def context_feature_columns():
  return {}

def example_feature_columns():
  sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="document_tokens", hash_bucket_size=100)
  document_embedding = tf.feature_column.embedding_column(
      categorical_column=sparse_column, dimension=20)
  return {"document_tokens": document_embedding}

def scoring_function(context_features, example_features, mode):
  """A feed-forward network to score query-document pairs."""
  input_layer = [
      tf.compat.v1.layers.flatten(context_features["document_tokens"])
  ]
  cur_layer = tf.concat(input_layer, 1)
  for dim in [256, 128, 64, 32]:
    cur_layer = tf.compat.v1.layers.dense(cur_layer, units=dim)
    cur_layer = tf.nn.relu(cur_layer)
  return tf.compat.v1.layers.dense(cur_layer, units=1)

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
      model_dir="/path/to/your/model/directory")            # step 1

  ranking_estimator = tfr.estimator.EstimatorBuilder(       # step 2
        context_feature_columns(),
        example_feature_columns(),
        scoring_function=scoring_function,
        hparams=hparams).make_estimator()

  ranking_pipeline = tfr.ext.pipeline.RankingPipeline(      # step 3
        context_feature_columns(),
        example_feature_columns(),
        hparams=hparams,
        estimator=ranking_estimator,
        label_feature_name="utility",
        label_feature_type=tf.float32)
  ranking_pipeline.train_and_eval()
```
