<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.feature" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.feature

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/feature.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Feature transformations for ranking in Keras.

## Classes

[`class EncodeListwiseFeatures`](../../tfr/keras/feature/EncodeListwiseFeatures.md):
A layer that produces dense `Tensors` from context and example features.

[`class GenerateMask`](../../tfr/keras/feature/GenerateMask.md): Layer to
generate mask.

## Functions

[`create_keras_inputs(...)`](../../tfr/keras/feature/create_keras_inputs.md):
Create Keras inputs from context and example feature columns.

[`deserialize_feature_columns(...)`](../../tfr/keras/feature/deserialize_feature_columns.md):
Deserializes dict of feature column configs.

[`serialize_feature_columns(...)`](../../tfr/keras/feature/serialize_feature_columns.md):
Serializes feature columns to a dict of class name and config.
