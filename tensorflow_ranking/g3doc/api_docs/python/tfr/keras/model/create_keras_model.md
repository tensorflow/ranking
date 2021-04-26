description: Creates a Functional Keras ranking model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model.create_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.model.create_keras_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py#L17-L64">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates a Functional Keras ranking model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.model.create_keras_model(
    network, loss, metrics, optimizer, size_feature_name, list_size=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

A mask is inferred from size_feature_name and passed to the network, along with
feature dictionary as inputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`network`
</td>
<td>
(<a href="../../../tfr/keras/network/RankingNetwork.md"><code>tfr.keras.network.RankingNetwork</code></a>) A ranking network which
generates a list of scores.
</td>
</tr><tr>
<td>
`loss`
</td>
<td>
(`tfr.keras.losses._RankingLoss`) A ranking loss.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
(list) List of ranking metrics, `tfr.keras.metrics._RankingMetric`
instances.
</td>
</tr><tr>
<td>
`optimizer`
</td>
<td>
(`tf.keras.optimizer.Optimizer`) Optimizer to minimize ranking
loss.
</td>
</tr><tr>
<td>
`size_feature_name`
</td>
<td>
(str) Name of feature for example list sizes. If not
None, this feature name corresponds to a `tf.int32` Tensor of size
[batch_size] corresponding to sizes of example lists. If `None`, all
examples are treated as valid.
</td>
</tr><tr>
<td>
`list_size`
</td>
<td>
(int) The list size for example features. If None, use dynamic
list size. A fixed list size is required for TPU training.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A compiled ranking Keras model, a `tf.keras.Model` instance.
</td>
</tr>

</table>
