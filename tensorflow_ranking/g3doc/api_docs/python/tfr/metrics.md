<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.metrics

Defines ranking metrics as TF ops.

Defined in
[`python/metrics.py`](https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py).

<!-- Placeholder for "Used in" -->

The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.

## Classes

[`class RankingMetricKey`](../tfr/metrics/RankingMetricKey.md): Ranking metric
key strings.

## Functions

[`average_relevance_position(...)`](../tfr/metrics/average_relevance_position.md):
Computes average relevance position (ARP).

[`discounted_cumulative_gain(...)`](../tfr/metrics/discounted_cumulative_gain.md):
Computes discounted cumulative gain (DCG).

[`make_ranking_metric_fn(...)`](../tfr/metrics/make_ranking_metric_fn.md):
Factory method to create a ranking metric function.

[`mean_reciprocal_rank(...)`](../tfr/metrics/mean_reciprocal_rank.md): Computes
mean reciprocal rank (MRR).

[`normalized_discounted_cumulative_gain(...)`](../tfr/metrics/normalized_discounted_cumulative_gain.md):
Computes normalized discounted cumulative gain (NDCG).

[`ordered_pair_accuracy(...)`](../tfr/metrics/ordered_pair_accuracy.md):
Computes the percentage of correctedly ordered pair.

[`precision(...)`](../tfr/metrics/precision.md): Computes precision as weighted
average of relevant examples.
