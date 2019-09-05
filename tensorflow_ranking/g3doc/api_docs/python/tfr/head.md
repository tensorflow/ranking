<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.head" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.head

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/head.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>

Defines `Head`s of TF ranking models.

<!-- Placeholder for "Used in" -->

Given logits (or output of a hidden layer), a `Head` computes predictions, loss,
train_op, metrics and exports outputs.

## Classes

[`class LossSpec`](../tfr/head/LossSpec.md): LossSpec(training_loss,
unreduced_loss, weights, processed_labels)

## Functions

[`create_ranking_head(...)`](../tfr/head/create_ranking_head.md): A factory
method to create `_RankingHead`.
