description: Orbit task for TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.task" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DATASET_FN_MAP"/>
<meta itemprop="property" content="MASK"/>
</div>

# Module: tfr.extension.task

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Orbit task for TF-Ranking.

Note: First - These APIs require These APS require the
`tensorflow_models`package. You can install it with `pip install
tf-models-official`. Second - Nothing under
`tensorflow_ranking.extension.premade` is imported by default. To use these APIs
import `premade` in your code: `import tensorflow_ranking.extension.premade`.

## Classes

[`class RankingDataConfig`](../../tfr/extension/task/RankingDataConfig.md): Data
set config.

[`class RankingDataLoader`](../../tfr/extension/task/RankingDataLoader.md): A
class to load dataset for ranking task.

[`class RankingTask`](../../tfr/extension/task/RankingTask.md): Task object for
TF-Ranking.

[`class RankingTaskConfig`](../../tfr/extension/task/RankingTaskConfig.md): The
TF-Ranking task config.

## Type Aliases

[`FeatureSpec`](../../tfr/extension/task/FeatureSpec.md)

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
