description: Defines Keras Layers for TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.layers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.layers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/layers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Defines Keras Layers for TF-Ranking.

## Classes

[`class ConcatFeatures`](../../tfr/keras/layers/ConcatFeatures.md): Concatenates
context features and example features in a listwise manner.

[`class DocumentInteractionAttention`](../../tfr/keras/layers/DocumentInteractionAttention.md):
Cross Document Interaction Attention layer.

[`class FlattenList`](../../tfr/keras/layers/FlattenList.md): Layer to flatten
the example list.

[`class GAMLayer`](../../tfr/keras/layers/GAMLayer.md): Defines a generalized
additive model (GAM) layer.

[`class RestoreList`](../../tfr/keras/layers/RestoreList.md): Output layer to
restore listwise output shape.

## Functions

[`create_tower(...)`](../../tfr/keras/layers/create_tower.md): Creates a
feed-forward network as `tf.keras.Sequential`.
