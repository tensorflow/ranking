description: Orbit task for TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.task" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DATASET_FN_MAP"/>
<meta itemprop="property" content="MASK"/>
</div>

# Module: tfr.keras.task

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/task.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Orbit task for TF-Ranking.

## Classes

[`class RankingDataConfig`](../../tfr/keras/task/RankingDataConfig.md): Data set
config.

[`class RankingDataLoader`](../../tfr/keras/task/RankingDataLoader.md): A class
to load dataset for ranking task.

[`class RankingTask`](../../tfr/keras/task/RankingTask.md): Task object for
TF-Ranking.

[`class RankingTaskConfig`](../../tfr/keras/task/RankingTaskConfig.md): The
TF-Ranking task config.

## Type Aliases

[`FeatureSpec`](../../tfr/keras/task/FeatureSpec.md): The central part of
internal API.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
DATASET_FN_MAP<a id="DATASET_FN_MAP"></a>
</td>
<td>
`{
 'textline': <class 'tensorflow.python.data.ops.readers.TextLineDatasetV2'>,
 'tfrecord': <class 'tensorflow.python.data.ops.readers.TFRecordDatasetV2'>
}`
</td>
</tr><tr>
<td>
MASK<a id="MASK"></a>
</td>
<td>
`'example_list_mask'`
</td>
</tr>
</table>
