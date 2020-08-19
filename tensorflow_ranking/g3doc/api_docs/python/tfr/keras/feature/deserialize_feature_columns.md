<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.feature.deserialize_feature_columns" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.feature.deserialize_feature_columns

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Deserializes dict of feature column configs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.feature.deserialize_feature_columns(
    feature_column_configs, custom_objects=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`feature_column_configs`
</td>
<td>
(dict) A dict mapping feature names to Keras feature
column config, could be generated using `serialize_feature_columns`.
</td>
</tr><tr>
<td>
`custom_objects`
</td>
<td>
(dict) Optional dictionary mapping names to custom classes
or functions to be considered during deserialization.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict mapping feature names to feature columns.
</td>
</tr>

</table>
