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

## `TFR-BERT` Extension

To incorporate the state-of-the-art pretrained BERT models for ranking problems,
we introduce TFR-BERT --- a generic framework (in TF-Ranking) that constructs
Learning-to-Ranking (LTR) models through finetuning the BERT representation of
query-document pairs.

In TFR-BERT, queries and documents are firstly encoded using BERT, and on top of
that a LTR model constructed in TF-Ranking is applied to further optimize the
ranking performance. With this architecture, we can easily fine-tune the ranking
models with a variety of pointwise, pairwise and listwise LTR methods that are
readily available in TF-Ranking. For more details, one may refer to our
[arxiv paper](https://arxiv.org/pdf/2004.08476.pdf).

Note that TFR-BERT also works for ranking applications besides the query-based
document ranking, as long as there are textual features. Hereafter, we will keep
using query and document for simplicity.

To create a TFR-BERT client, one needs to follow the steps below.

### Installing TFR-BERT

TFR-BERT is included in the latest TF-Ranking PyPI package. You can simply
upgrade by following the instruction
[here](https://github.com/tensorflow/ranking#stable-builds).

### Preparing Data

TFR-BERT requires the data being stored with BERT convention. In particular, we
treat each query-document pair as two sentences, and convert them into the
following format:

`[CLS] query text [SEP] document text [SEP]`

Here, [CLS] indicates the start of a query-document pair, and [SEP] denotes the
separator between query and document. We also truncate the whole sequence text
if it exceeds 512 tokens (the maximum sequence length limit in BERT).

Then, query and document are tokenized, and mapped to BERT word ids. One may use
the BERT
[FullTokenizer](https://github.com/google-research/bert/blob/master/tokenization.py#L161).
TFR-BERT also provides an utility function for such a purpose. Please see more
details on
[`TFRBertUtil._to_bert_ids()`](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/tfrbert.py).

To use TF-Ranking, you also need to convert query and candidate documents into
the ELWC proto format (`tensorflow.serving.ExampleListWithContext`), in which a
list of documents (examples) are wrapped together for a given query (context).
One can use the
[`TFRBertUtil.convert_to_elwc()`](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/tfrbert.py)
for such a conversion.

### Creating a TFR-BERT Model

We have built the `TFRBertRankingNetwork` and `RankingPipeline` components which
greatly simplify the construction of a TFR-BERT client. A complete example can
be found in
[`bert_ranking_example.py`](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/examples/tfrbert_example.py).
Here, we will provide a detailed explanation.

#### Step 1: Defining `hparams`

To use `TFRBertRankingNetwork` and `RankingPipeline`, one needs to define a set
of hparams --- a python dictionary that stores all of the parameters. The
definitin for each parameter can be found
[here](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/extension/README.md#step-1-defining-hparams).
In addition to TF-Ranking parameters, one also needs to specify the BERT-related
parameters such as `bert_config_file`, `bert_init_ckpt`, `bert_max_seq_length`,
and `bert_num_warmup_steps`.

*   The `bert_config_file` and `bert_init_ckpt` requires TF2 checkpoints and can
    be directly downloaded from
    [TF Models website](https://github.com/tensorflow/models/tree/master/official/nlp/bert).
    If you have an older version of checkpoint, you can convert it to TF2
    version by
    [this script](https://github.com/tensorflow/models/blob/master/official/nlp/bert/tf2_encoder_checkpoint_converter.py).
*   The `bert_max_seq_length` refers to maximum input sequence length (#words)
    after WordPiece tokenization. Sequences longer than this will be truncated,
    and sequences shorter than this will be padded.
*   The `bert_num_warmup_steps` will be used for adjusting the learning rate. If
    global_step < warmup_steps, learning rate will be
    `global_step/warmup_steps * init_lr`. This was implemented in the
    bert/optimization.py file.

```python
hparams = dict(
      train_input_pattern="/path/to/train/files",
      eval_input_pattern="/path/to/eval/files",
      learning_rate=1e-5,
      train_batch_size=8,
      eval_batch_size=8,
      checkpoint_secs=300, # every five minutes
      num_checkpoints=100,
      num_train_steps=100000,
      num_eval_steps=100,
      loss="softmax_loss",
      list_size=5,
      listwise_inference=False,
      convert_labels_to_binary=False,
      model_dir="/path/to/your/model/directory",
      bert_config_file="/path/to/bert/config/file.json",
      bert_init_ckpt="/path/to/bert/init.ckpt",
      bert_max_seq_length=64,
      bert_num_warmup_steps=10000)
```

#### Step 2: Building a TFRBertRankingNetwork

The next step is to define a TF-Ranking model to score each query and document
pair. With the help from `TFRBertRankingNetwork`, we only need to define the
`context_feature_columns`, the `example_feature_columns`, as well as the scoring
logic for each query and document pair.

```python
from tensorflow_ranking import tfr
from tfr.extension import tfrbert

def context_feature_columns():
  return {}


def example_feature_columns():
  bert_max_seq_length = 64
  return {
      "input_ids":
          tf.feature_column.numeric_column(
              "input_ids",
              shape=(bert_max_seq_length,),
              default_value=0,
              dtype=tf.int64),
      "input_mask":
          tf.feature_column.numeric_column(
              "input_mask",
              shape=(bert_max_seq_length,),
              default_value=0,
              dtype=tf.int64),
      "segment_ids":
          tf.feature_column.numeric_column(
              "segment_ids",
              shape=(bert_max_seq_length,),
              default_value=0,
              dtype=tf.int64),
  }


def get_estimator(hparams):
  util = tfrbert.TFRBertUtil(
      bert_config_file=hparams.get("bert_config_file"),
      bert_init_ckpt=hparams.get("bert_init_ckpt"),
      bert_max_seq_length=hparams.get("bert_max_seq_length"))

  network = tfrbert.TFRBertRankingNetwork(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      bert_config_file=hparams.get("bert_config_file"),
      bert_max_seq_length=hparams.get("bert_max_seq_length"),
      bert_output_dropout=hparams.get("dropout_rate"),
      name="tfrbert")

  loss = tfr.keras.losses.get(
      hparams.get("loss"),
      reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)

  metrics = tfr.keras.metrics.default_keras_metrics()

  config = tf.estimator.RunConfig(
      model_dir=hparams.get("model_dir"),
      keep_checkpoint_max=hparams.get("num_checkpoints"),
      save_checkpoints_secs=hparams.get("checkpoint_secs"))

  optimizer = util.create_optimizer(
      init_lr=hparams.get("learning_rate"),
      train_steps=hparams.get("num_train_steps"),
      warmup_steps=hparams.get("bert_num_warmup_steps"))

  ranker = tfr.keras.model.create_keras_model(
      network=network,
      loss=loss,
      metrics=metrics,
      optimizer=optimizer,
      size_feature_name="example_list_size")

  return tfr.keras.estimator.model_to_estimator(
      model=ranker,
      model_dir=hparams.get("model_dir"),
      config=config,
      warm_start_from=util.get_warm_start_settings(exclude="tfrbert"))
```

#### Step 3: Constructing a `RankingPipeline`

With all of the above components, one can define a `RankingPipeline` instance,
and simply call the `train_and_eval()` function for model training and eval.

```python
import tensorflow_ranking as tfr

bert_ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
    context_feature_columns=context_feature_columns(),
    example_feature_columns=example_feature_columns(),
    hparams=hparams,
    estimator=get_estimator(hparams),
    label_feature_name="relevance",
    label_feature_type=tf.int64,
    size_feature_name="example_list_size")

bert_ranking_pipeline.train_and_eval()
```

### Running the TFR-BERT Example

You can now run our `tfrbert_example.py` example. To do so, please download a
BERT checkpoint from
[here](https://github.com/tensorflow/models/tree/master/official/nlp/bert), and
the Antique dataset
[here](http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/). This dataset has
already been converted to the ELWC proto format, and stored in the TFRecord
format.

Specifically, each BERT checkpoint contains four files: `bert_model.ckpt.index`,
`bert_model.ckpt.data-00000-of-00001`, `vocab.txt` and `bert_config.json`. When
defining `bert_init_ckpt` (see below), you can directly use `bert_model.ckpt`.
It reads both the data file and index file. As for the Antique dataset, please
make sure it includes both `train-pair.tdrecords` and `test-pair.tdrecords`.

Note that for testing purposes, you may want to try `BERT-Tiny` or `BERT-Mini`,
which are sufficiently small and fast for CPU use. If you want to use a large
BERT model, please use GPU and make sure you have sufficient memory. In case of
running out of memory, you can either reduce `list_size` or try a different BERT
checkpoint.

After placing those files to your local folders, you may run the command below.

```bash
  MODEL_DIR=/tmp/output && \
  TRAIN=path/to/your/train-pair.tfrecords && \
  TEST=path/to/your/test-pair.tfrecords && \
  rm -rf $MODEL_DIR && \
  bazel build -c opt \
  tensorflow_ranking/extension/examples/tfrbert_example_py_binary && \
  ./bazel-bin/tensorflow_ranking/extension/examples/tfrbert_example_py_binary \
    --train_input_pattern=${TRAIN} \
    --eval_input_pattern=${TEST} \
    --bert_config_file=<bert config, ending with .json> \
    --bert_init_ckpt=<bert checkpoint, ending with .ckpt> \
    --bert_max_seq_length=128 \
    --model_dir=${MODEL_DIR} \
    --list_size=50 \
    --train_batch_size=8 \
    --learning_rate=1e-04 \
    --num_train_steps=100000 \
    --checkpoint_secs=120 \
    --num_checkpoints=10 \
    --alsologtostderr
```
