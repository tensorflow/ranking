description: A class to load dataset for TFR-BERT task.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.premade.TFRBertDataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="load"/>
</div>

# tfr.keras.premade.TFRBertDataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/premade/tfrbert_task.py#L36-L83">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A class to load dataset for TFR-BERT task.

Inherits From:
[`RankingDataLoader`](../../../tfr/keras/task/RankingDataLoader.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfr.keras.premade.tfrbert_task.TFRBertDataLoader`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.premade.TFRBertDataLoader(
    params,
    label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="load"><code>load</code></h3>

<a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/task.py#L129-L139">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load(
    input_context: Optional[tf.distribute.InputContext] = None
) -> tf.data.Dataset
</code></pre>

Returns a tf.dataset.Dataset.
