<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.losses" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: ranking.losses



Defined in [`python/losses.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/losses.py).

Defines ranking losses as TF ops.

The losses here are used to learn TF ranking models. It works with listwise
Tensors only.<!-- Placeholder for "Used in" -->



## Classes

[`class DCGLambdaWeight`](../ranking/losses/DCGLambdaWeight.md): LambdaWeight for Discounted Cumulative Gain metric.

[`class ListMLELambdaWeight`](../ranking/losses/ListMLELambdaWeight.md): LambdaWeight for ListMLE cost function.

[`class PrecisionLambdaWeight`](../ranking/losses/PrecisionLambdaWeight.md): LambdaWeight for Precision metric.

[`class RankingLossKey`](../ranking/losses/RankingLossKey.md): Ranking loss key strings.

## Functions

[`create_ndcg_lambda_weight(...)`](../ranking/losses/create_ndcg_lambda_weight.md): Creates _LambdaWeight for NDCG metric.

[`create_p_list_mle_lambda_weight(...)`](../ranking/losses/create_p_list_mle_lambda_weight.md): Creates _LambdaWeight based on Position-Aware ListMLE paper.

[`create_reciprocal_rank_lambda_weight(...)`](../ranking/losses/create_reciprocal_rank_lambda_weight.md): Creates _LambdaWeight for MRR-like metric.

[`make_loss_fn(...)`](../ranking/losses/make_loss_fn.md): Makes a loss function using a single loss or multiple losses.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

