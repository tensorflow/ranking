description: A class to load dataset for ranking task.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.extension.task.RankingDataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="load"/>
</div>

# tfr.extension.task.RankingDataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L78-L155">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A class to load dataset for ranking task.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.extension.task.RankingDataLoader(
    params,
    context_feature_spec: <a href="../../../tfr/extension/task/FeatureSpec.md"><code>tfr.extension.task.FeatureSpec</code></a> = None,
    example_feature_spec: <a href="../../../tfr/extension/task/FeatureSpec.md"><code>tfr.extension.task.FeatureSpec</code></a> = None,
    label_spec: Tuple[str, tf.io.FixedLenFeature] = None,
    dataset_fn: Optional[Callable[[], tf.data.Dataset]] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="load"><code>load</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension/task.py#L145-L155">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load(
    input_context: Optional[tf.distribute.InputContext] = None
) -> tf.data.Dataset
</code></pre>

Returns a tf.dataset.Dataset.
