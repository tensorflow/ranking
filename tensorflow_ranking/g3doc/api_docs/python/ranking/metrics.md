<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="ranking.metrics" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: ranking.metrics



Defined in [`python/metrics.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py).

Defines ranking metrics as TF ops.

The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.<!-- Placeholder for "Used in" -->



## Classes

[`class RankingMetricKey`](../ranking/metrics/RankingMetricKey.md): Ranking metric key strings.

## Functions

[`average_relevance_position(...)`](../ranking/metrics/average_relevance_position.md): Computes average relevance position (ARP).

[`discounted_cumulative_gain(...)`](../ranking/metrics/discounted_cumulative_gain.md): Computes discounted cumulative gain (DCG).

[`make_ranking_metric_fn(...)`](../ranking/metrics/make_ranking_metric_fn.md): Factory method to create a ranking metric function.

[`mean_reciprocal_rank(...)`](../ranking/metrics/mean_reciprocal_rank.md): Computes mean reciprocal rank (MRR).

[`normalized_discounted_cumulative_gain(...)`](../ranking/metrics/normalized_discounted_cumulative_gain.md): Computes normalized discounted cumulative gain (NDCG).

[`ordered_pair_accuracy(...)`](../ranking/metrics/ordered_pair_accuracy.md): Computes the percentage of correctedly ordered pair.

[`precision(...)`](../ranking/metrics/precision.md): Computes precision as weighted average of relevant examples.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

