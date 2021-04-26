description: Ranking pipeline to train tf.keras.Model in TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.pipeline" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.pipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/pipeline.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Ranking pipeline to train tf.keras.Model in TF-Ranking.

## Classes

[`class AbstractDatasetBuilder`](../../tfr/keras/pipeline/AbstractDatasetBuilder.md):
Interface for datasets and signatures.

[`class AbstractPipeline`](../../tfr/keras/pipeline/AbstractPipeline.md):
Defines the interface for a pipeline to train a ranking tf.keras.Model.

[`class BaseDatasetBuilder`](../../tfr/keras/pipeline/BaseDatasetBuilder.md):
Builds datasets from feature specs.

[`class DatasetHparams`](../../tfr/keras/pipeline/DatasetHparams.md): Hparams
used in dataset_builder.

[`class ModelFitPipeline`](../../tfr/keras/pipeline/ModelFitPipeline.md):
Pipeline using model.fit to train a ranking tf.keras.Model.

[`class MultiLabelDatasetBuilder`](../../tfr/keras/pipeline/MultiLabelDatasetBuilder.md):
Builds datasets for multi-task training.

[`class MultiTaskPipeline`](../../tfr/keras/pipeline/MultiTaskPipeline.md):
Pipeline for multi-task training.

[`class NullDatasetBuilder`](../../tfr/keras/pipeline/NullDatasetBuilder.md): An
no-op wrapper of datasets and signatures.

[`class PipelineHparams`](../../tfr/keras/pipeline/PipelineHparams.md): Hparams
used in pipeline.

[`class SimpleDatasetBuilder`](../../tfr/keras/pipeline/SimpleDatasetBuilder.md):
Builds datasets from feature specs with a single label spec.

[`class SimplePipeline`](../../tfr/keras/pipeline/SimplePipeline.md): Pipleine
for single-task training.
