<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.estimator.model_to_estimator" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.estimator.model_to_estimator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/estimator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Keras ranking model to Estimator.

```python
tfr.keras.estimator.model_to_estimator(
    model, model_dir=None, config=None
)
```

<!-- Placeholder for "Used in" -->

This function is based on the custom model_fn in TF2.0 migration guide.
https://www.tensorflow.org/guide/migrate#custom_model_fn_with_tf_20_symbols

#### Args:

*   <b>`model`</b>: (tf.keras.Model) A ranking keras model, which can be created
    using
    <a href="../../../tfr/keras/model/create_keras_model.md"><code>tfr.keras.model.create_keras_model</code></a>.
    Masking is handled inside this function.
*   <b>`model_dir`</b>: (str) Directory to save `Estimator` model graph and
    checkpoints.
*   <b>`config`</b>: (tf.estimator.RunConfig) Specified config for distributed
    training and checkpointing.

#### Returns:

(tf.estimator.Estimator) A ranking estimator.
