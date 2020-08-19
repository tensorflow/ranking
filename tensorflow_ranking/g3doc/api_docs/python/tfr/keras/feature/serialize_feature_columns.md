<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.feature.serialize_feature_columns" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.feature.serialize_feature_columns

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Serializes feature columns to a dict of class name and config.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.feature.serialize_feature_columns(
    feature_columns
)
</code></pre>

<!-- Placeholder for "Used in" -->

This serialization is required to support for SavedModel using model.save() in
Keras. The serialization is similar to that of `tf.keras.layers.DenseFeatures`,
which also has feature columns in it's constructor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`feature_columns`
</td>
<td>
(dict) feature names to feature columns.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict mapping feature names to serialized feature columns.
</td>
</tr>

</table>
