<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.feature.encode_features" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.feature.encode_features

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Returns dense tensors from features using feature columns.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.feature.encode_features(
    features, feature_columns, mode=tf.estimator.ModeKeys.TRAIN, scope=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This function encodes the feature column transformation on the 'raw' `features`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`features`
</td>
<td>
(dict) mapping feature names to feature values, possibly obtained
from input_fn.
</td>
</tr><tr>
<td>
`feature_columns`
</td>
<td>
(list)  list of feature columns.
</td>
</tr><tr>
<td>
`mode`
</td>
<td>
(`estimator.ModeKeys`) Specifies if this is training, evaluation or
inference. See `ModeKeys`.
</td>
</tr><tr>
<td>
`scope`
</td>
<td>
(str) variable scope for the per column input layers.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
(dict) A mapping from columns to dense tensors.
</td>
</tr>

</table>
