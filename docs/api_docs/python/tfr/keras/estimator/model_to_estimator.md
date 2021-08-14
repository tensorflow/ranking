description: Keras ranking model to Estimator.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.estimator.model_to_estimator" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.estimator.model_to_estimator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/estimator.py#L16-L145">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Keras ranking model to Estimator.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.estimator.model_to_estimator(
    model, model_dir=None, config=None, custom_objects=None,
    weights_feature_name=None, warm_start_from=None,
    serving_default=&#x27;regress&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->

This function is based on the custom model_fn in TF2.0 migration guide.
https://www.tensorflow.org/guide/migrate#custom_model_fn_with_tf_20_symbols

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`
</td>
<td>
(tf.keras.Model) A ranking keras model, which  can be created using
<a href="../../../tfr/keras/model/create_keras_model.md"><code>tfr.keras.model.create_keras_model</code></a>. Masking is handled inside this
function.
</td>
</tr><tr>
<td>
`model_dir`
</td>
<td>
(str) Directory to save `Estimator` model graph and checkpoints.
</td>
</tr><tr>
<td>
`config`
</td>
<td>
(tf.estimator.RunConfig) Specified config for distributed training
and checkpointing.
</td>
</tr><tr>
<td>
`custom_objects`
</td>
<td>
(dict) mapping names (strings) to custom objects (classes
and functions) to be considered during deserialization.
</td>
</tr><tr>
<td>
`weights_feature_name`
</td>
<td>
(str) A string specifying the name of the per-example
(of shape [batch_size, list_size]) or per-list (of shape [batch_size, 1])
weights feature in `features` dict.
</td>
</tr><tr>
<td>
`warm_start_from`
</td>
<td>
(`tf.estimator.WarmStartSettings`) settings to warm-start
the `tf.estimator.Estimator`.
</td>
</tr><tr>
<td>
`serving_default`
</td>
<td>
(str) Specifies "regress" or "predict" as the
serving_default signature.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
(tf.estimator.Estimator) A ranking estimator.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if weights_feature_name is not in features.
</td>
</tr>
</table>
