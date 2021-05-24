description: Ranking model utilities and classes in tfr.keras.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.model" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/model.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Ranking model utilities and classes in tfr.keras.

## Classes

[`class AbstractModelBuilder`](../../tfr/keras/model/AbstractModelBuilder.md):
Interface to build a `tf.keras.Model` for ranking.

[`class DNNScorer`](../../tfr/keras/model/DNNScorer.md): Univariate scorer using
DNN.

[`class FeatureSpecInputCreator`](../../tfr/keras/model/FeatureSpecInputCreator.md):
InputCreator with feature specs.

[`class GAMScorer`](../../tfr/keras/model/GAMScorer.md): Univariate scorer using
GAM.

[`class InputCreator`](../../tfr/keras/model/InputCreator.md): Interface for
input creator.

[`class ModelBuilder`](../../tfr/keras/model/ModelBuilder.md): Builds a
`tf.keras.Model`.

[`class ModelBuilderWithMask`](../../tfr/keras/model/ModelBuilderWithMask.md):
Interface to build a `tf.keras.Model` for ranking with a mask Tensor.

[`class Preprocessor`](../../tfr/keras/model/Preprocessor.md): Interface for
feature preprocessing.

[`class PreprocessorWithSpec`](../../tfr/keras/model/PreprocessorWithSpec.md):
Preprocessing inputs with provided spec.

[`class Scorer`](../../tfr/keras/model/Scorer.md): Interface for scorer.

[`class TypeSpecInputCreator`](../../tfr/keras/model/TypeSpecInputCreator.md):
InputCreator with tensor type specs.

[`class UnivariateScorer`](../../tfr/keras/model/UnivariateScorer.md): Interface
for univariate scorer.

## Functions

[`create_keras_model(...)`](../../tfr/keras/model/create_keras_model.md):
Creates a Functional Keras ranking model.

## Type Aliases

[`TensorLike`](../../tfr/keras/model/TensorLike.md)
