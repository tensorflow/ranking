description: TensorFlow Ranking Premade Orbit Task Module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.premade" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DOCUMENT_ID"/>
<meta itemprop="property" content="QUERY_ID"/>
</div>

# Module: tfr.extension.premade

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/premade/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

TensorFlow Ranking Premade Orbit Task Module.

Note: First - These APIs require These APS require the
`tensorflow_models`package. You can install it with `pip install
tf-models-official`. Second - Nothing under
`tensorflow_ranking.extension.premade` is imported by default. To use these APIs
import `premade` in your code: `import tensorflow_ranking.extension.premade`.

## Modules

[`tfrbert_task`](../../tfr/extension/premade/tfrbert_task.md) module: TF-Ranking
BERT task.

## Classes

[`class TFRBertConfig`](../../tfr/extension/premade/TFRBertConfig.md): The
tf-ranking BERT task config.

[`class TFRBertDataConfig`](../../tfr/extension/premade/TFRBertDataConfig.md):
Data config for TFR-BERT task.

[`class TFRBertDataLoader`](../../tfr/extension/premade/TFRBertDataLoader.md): A
class to load dataset for TFR-BERT task.

[`class TFRBertModelBuilder`](../../tfr/extension/premade/TFRBertModelBuilder.md):
Model builder for TFR-BERT models.

[`class TFRBertModelConfig`](../../tfr/extension/premade/TFRBertModelConfig.md):
A TFR-BERT model configuration.

[`class TFRBertScorer`](../../tfr/extension/premade/TFRBertScorer.md):
Univariate BERT-based scorer.

[`class TFRBertTask`](../../tfr/extension/premade/TFRBertTask.md): Task object
for tf-ranking BERT.

## Type Aliases

[`TensorDict`](../../tfr/extension/premade/TensorDict.md)

[`TensorLike`](../../tfr/keras/model/TensorLike.md)

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
DOCUMENT_ID<a id="DOCUMENT_ID"></a>
</td>
<td>
`'document_id'`
</td>
</tr><tr>
<td>
QUERY_ID<a id="QUERY_ID"></a>
</td>
<td>
`'query_id'`
</td>
</tr>
</table>
