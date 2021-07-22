description: A class to load dataset for ranking task.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.task.RankingDataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="load"/>
</div>

# tfr.keras.task.RankingDataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/task.py#L62-L139">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A class to load dataset for ranking task.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.task.RankingDataLoader(
    params,
    context_feature_spec: <a href="../../../tfr/keras/task/FeatureSpec.md"><code>tfr.keras.task.FeatureSpec</code></a> = None,
    example_feature_spec: <a href="../../../tfr/keras/task/FeatureSpec.md"><code>tfr.keras.task.FeatureSpec</code></a> = None,
    label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
    dataset_fn: Optional[Callable[[], tf.data.Dataset]] = None
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
