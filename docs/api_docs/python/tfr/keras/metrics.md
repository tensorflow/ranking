description: Keras metrics in TF-Ranking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.metrics" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfr.keras.metrics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Keras metrics in TF-Ranking.

NOTE: For metrics that compute a ranking, ties are broken randomly. This means
that metrics may be stochastic if items with equal scores are provided.

WARNING: Some metrics (e.g. Recall or MRR) are not well-defined when there are
no relevant items (e.g. if `y_true` has a row of only zeroes). For these cases,
the TF-Ranking metrics will evaluate to `0`.

## Classes

[`class ARPMetric`](../../tfr/keras/metrics/ARPMetric.md): Average relevance
position (ARP).

[`class AlphaDCGMetric`](../../tfr/keras/metrics/AlphaDCGMetric.md): Alpha
discounted cumulative gain (alphaDCG).

[`class DCGMetric`](../../tfr/keras/metrics/DCGMetric.md): Discounted cumulative
gain (DCG).

[`class MRRMetric`](../../tfr/keras/metrics/MRRMetric.md): Mean reciprocal rank
(MRR).

[`class MeanAveragePrecisionMetric`](../../tfr/keras/metrics/MeanAveragePrecisionMetric.md):
Mean average precision (MAP).

[`class NDCGMetric`](../../tfr/keras/metrics/NDCGMetric.md): Normalized
discounted cumulative gain (NDCG).

[`class OPAMetric`](../../tfr/keras/metrics/OPAMetric.md): Ordered pair accuracy
(OPA).

[`class PrecisionIAMetric`](../../tfr/keras/metrics/PrecisionIAMetric.md):
Precision-IA@k (Pre-IA@k).

[`class PrecisionMetric`](../../tfr/keras/metrics/PrecisionMetric.md):
Precision@k (P@k).

[`class RankingMetricKey`](../../tfr/keras/metrics/RankingMetricKey.md): Ranking
metric key strings.

[`class RecallMetric`](../../tfr/keras/metrics/RecallMetric.md): Recall@k (R@k).

## Functions

[`default_keras_metrics(...)`](../../tfr/keras/metrics/default_keras_metrics.md):
Returns a list of ranking metrics.

[`get(...)`](../../tfr/keras/metrics/get.md): Factory method to get a list of
ranking metrics.
