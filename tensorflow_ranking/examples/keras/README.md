# TF-Ranking Keras User Guide

This guide outlines a step-by-step process for building and training Keras
ranking models.

[TOC]

We propose a simple and intuitive user journey to build ranking models using
Keras APIs.

As with the Estimator user journey, we define context and example feature
columns, and create a ranking dataset.

## Define Context and Example Feature Columns

```python
def create_feature_columns():
  sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="query_tokens", hash_bucket_size=100)
  query_embedding = tf.feature_column.embedding_column(
      categorical_column=sparse_column, dimension=20)
  context_feature_columns = {"query_tokens": query_embedding}

  sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="document_tokens", hash_bucket_size=100)
  document_embedding = tf.feature_column.embedding_column(
      categorical_column=sparse_column, dimension=20)
  example_feature_columns = {"document_tokens": document_embedding}

  return context_feature_columns, example_feature_columns
```

## Create Dataset using tfr.data APIs

```python
_SIZE = "example_list_size"  # Name of feature of example list sizes.

def make_dataset(file_pattern,
                 batch_size,
                 randomize_input=True,
                 num_epochs=None):
  context_feature_columns, example_feature_columns = create_feature_columns()
  context_feature_spec = tf.feature_column.make_parse_example_spec(
      context_feature_columns.values())
  label_column = tf.feature_column.numeric_column(
      _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
  example_feature_spec = tf.feature_column.make_parse_example_spec(
      list(example_feature_columns.values()) + [label_column])
  dataset = tfr.data.build_ranking_dataset(
      file_pattern=file_pattern,
      data_format=tfr.data.ELWC,
      batch_size=batch_size,
      context_feature_spec=context_feature_spec,
      example_feature_spec=example_feature_spec,
      reader=tf.data.TFRecordDataset,
      shuffle=randomize_input,
      num_epochs=num_epochs,
      size_feature_name=_SIZE)

  def _separate_features_and_label(features):
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)
    return features, label

  dataset = dataset.map(_separate_features_and_label)
  return dataset
```

## Define Ranking Network

Ranking Networks define the neural network architecture for scoring logic. The
output is a list of scores that can be used for loss and metric computation.

```python
context_feature_columns, example_feature_columns = create_feature_columns()
# Use a Premade Network, or subclass and build your own!
network = tfr.keras.canned.DNNRankingNetwork(
      context_feature_columns=context_feature_columns,
      example_feature_columns=example_feature_columns,
      hidden_layer_dims=[1024, 512, 256],
      activation=tf.nn.relu,
      dropout=0.5)
```

## Define Ranking Loss and Metric

Ranking losses and metrics can be easily defined and used. For losses, a factory
method is available.

```python
softmax_loss_obj = tfr.keras.losses.get(tfr.losses.RankingLossKey.SOFTMAX_LOSS)

# Contains all ranking metrics, including NDCG @ {1, 3, 5, 10}.
default_metrics = tfr.keras.metrics.get_default_metrics()
```

## Bring it all together: Create Ranking Model

```python
# Build ranker as a Functional Keras model.
ranker = tfr.keras.model.create_keras_model(
      network=network,
      loss=softmax_loss_obj,
      metrics=default_metrics,
      optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05),
      size_feature_name=_SIZE)
```

## Native Keras Training

These ranking models can be trained and evaluated natively using
`keras_model.fit()` and `keras_model.evaluate()`.

```python
ranker.fit(train_dataset,
           validation_data=vali_dataset,
           steps_per_epoch=1000,
           epochs=100,
           validation_steps=100)

ranker.evaluate(test_dataset)
```

## Training via Estimator APIs

Alternatively, these ranking models can be trained via Estimator training and
evaluation using `tfr.keras.model.model_to_estimator()`. Such a setup can be
compatible with existing production pipelines.

```python
estimator = tfr.keras.estimator.model_to_estimator(
      model=ranker, model_dir="/path/to/model/dir")

tf.estimator.parameterized_train_and_evaluate(
    train_and_eval_fn, hparams=hparams)
```

## Debugging

Keras additionally allows for model summaries and debugging.

```python
print(ranker.get_config())  # Inspect ranker’s parameters.

ranker.predict(dummy_example)  # Run prediction on a dummy example.
```
