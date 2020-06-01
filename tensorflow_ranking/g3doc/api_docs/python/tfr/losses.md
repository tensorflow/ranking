<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.losses" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.losses

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Defines ranking losses as TF ops.


The losses here are used to learn TF ranking models. It works with listwise
Tensors only.

## Classes

[`class RankingLossKey`](../tfr/losses/RankingLossKey.md): Ranking loss key
strings.

## Functions

[`create_ndcg_lambda_weight(...)`](../tfr/losses/create_ndcg_lambda_weight.md):
Creates _LambdaWeight for NDCG metric.

[`create_p_list_mle_lambda_weight(...)`](../tfr/losses/create_p_list_mle_lambda_weight.md):
Creates _LambdaWeight based on Position-Aware ListMLE paper.

[`create_reciprocal_rank_lambda_weight(...)`](../tfr/losses/create_reciprocal_rank_lambda_weight.md):
Creates _LambdaWeight for MRR-like metric.

[`make_loss_fn(...)`](../tfr/losses/make_loss_fn.md): Makes a loss function
using a single loss or multiple losses.

[`make_loss_metric_fn(...)`](../tfr/losses/make_loss_metric_fn.md): Factory
method to create a metric based on a loss.
