<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.feature.create_keras_inputs" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.feature.create_keras_inputs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Create Keras inputs from context and example feature columns.

```python
tfr.keras.feature.create_keras_inputs(
    context_feature_columns, example_feature_columns, size_feature_name
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`context_feature_columns`</b>: (dict) context feature names to columns.
*   <b>`example_feature_columns`</b>: (dict) example feature names to columns.
*   <b>`size_feature_name`</b>: (str) Name of feature for example list sizes. If
    not None, this feature name corresponds to a `tf.int32` Tensor of size
    [batch_size] corresponding to sizes of example lists.

#### Returns:

A dict mapping feature names to Keras Input tensors.
