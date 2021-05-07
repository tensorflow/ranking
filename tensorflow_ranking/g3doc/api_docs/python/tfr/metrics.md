description: Defines ranking metrics as TF ops.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.metrics" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.metrics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Defines ranking metrics as TF ops.


The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.

## Classes

[`class RankingMetricKey`](../tfr/metrics/RankingMetricKey.md): Ranking metric
key strings.

## Functions

[`alpha_discounted_cumulative_gain(...)`](../tfr/metrics/alpha_discounted_cumulative_gain.md):
Computes alpha discounted cumulative gain (alpha-DCG).

[`average_relevance_position(...)`](../tfr/metrics/average_relevance_position.md):
Computes average relevance position (ARP).

[`binary_preference(...)`](../tfr/metrics/binary_preference.md): Computes binary
preference (BPref).

[`compute_mean(...)`](../tfr/metrics/compute_mean.md): Returns the mean of the
specified metric given the inputs.

[`discounted_cumulative_gain(...)`](../tfr/metrics/discounted_cumulative_gain.md):
Computes discounted cumulative gain (DCG).

[`eval_metric(...)`](../tfr/metrics/eval_metric.md): A stand-alone method to
evaluate metrics on ranked results.

[`make_ranking_metric_fn(...)`](../tfr/metrics/make_ranking_metric_fn.md):
Factory method to create a ranking metric function.

[`mean_average_precision(...)`](../tfr/metrics/mean_average_precision.md):
Computes mean average precision (MAP).

[`mean_reciprocal_rank(...)`](../tfr/metrics/mean_reciprocal_rank.md): Computes
mean reciprocal rank (MRR).

[`normalized_discounted_cumulative_gain(...)`](../tfr/metrics/normalized_discounted_cumulative_gain.md):
Computes normalized discounted cumulative gain (NDCG).

[`ordered_pair_accuracy(...)`](../tfr/metrics/ordered_pair_accuracy.md):
Computes the percentage of correctly ordered pair.

[`precision(...)`](../tfr/metrics/precision.md): Computes precision as weighted
average of relevant examples.

[`precision_ia(...)`](../tfr/metrics/precision_ia.md): Computes Intent-Aware
Precision as weighted average of relevant examples.

[`recall(...)`](../tfr/metrics/recall.md): Computes recall as weighted average
of relevant examples.
