# Copyright 2021 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Keras metrics in TF-Ranking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional
import tensorflow.compat.v2 as tf

from tensorflow_ranking.python import metrics_impl
from tensorflow_ranking.python.keras import utils


class RankingMetricKey(object):
  """Ranking metric key strings."""
  # Mean Reciprocal Rank. For binary relevance.
  MRR = "mrr"

  # Average Relevance Position.
  ARP = "arp"

  # Normalized Discounted Cumulative Gain.
  NDCG = "ndcg"

  # Discounted Cumulative Gain.
  DCG = "dcg"

  # Precision. For binary relevance.
  PRECISION = "precision"

  # Mean Average Precision. For binary relevance.
  MAP = "map"

  # Intent-aware Precision. For binary relevance of subtopics.
  PRECISION_IA = "precision_ia"

  # Ordered Pair Accuracy.
  ORDERED_PAIR_ACCURACY = "ordered_pair_accuracy"

  # Alpha Discounted Cumulative Gain.
  ALPHA_DCG = "alpha_dcg"


def get(key: str,
        name: Optional[str] = None,
        dtype: Optional[tf.dtypes.DType] = None,
        topn: Optional[int] = None,
        **kwargs: Dict[str, Any]) -> tf.keras.metrics.Metric:
  """Factory method to get a list of ranking metrics.

  Example Usage:
  ```python
    metric = tfr.keras.metics.get(tfr.keras.metrics.RankingMetricKey.MRR)
  ```
  to get Mean Reciprocal Rank.
  ```python
    metric = tfr.keras.metics.get(tfr.keras.metrics.RankingMetricKey.MRR,
                                  topn=2)
  ```
  to get MRR@2.

  Args:
    key: An attribute of `RankingMetricKey`, defining which metric objects to
      return.
    name: Name of metrics.
    dtype: Dtype of the metrics.
    topn: Cutoff of how many items are considered in the metric.
    **kwargs: Keyword arguments for the metric object.

  Returns:
    A tf.keras.metrics.Metric. See `_RankingMetric` signature for more details.

  Raises:
    ValueError: If key is unsupported.
  """
  if not isinstance(key, str):
    raise ValueError("Input `key` needs to be string.")

  key_to_cls = {
      RankingMetricKey.MRR: MRRMetric,
      RankingMetricKey.ARP: ARPMetric,
      RankingMetricKey.PRECISION: PrecisionMetric,
      RankingMetricKey.MAP: MeanAveragePrecisionMetric,
      RankingMetricKey.NDCG: NDCGMetric,
      RankingMetricKey.DCG: DCGMetric,
      RankingMetricKey.ORDERED_PAIR_ACCURACY: OPAMetric,
  }

  metric_kwargs = {"name": name, "dtype": dtype}
  if topn:
    metric_kwargs.update({"topn": topn})

  if kwargs:
    metric_kwargs.update(kwargs)

  if key in key_to_cls:
    metric_cls = key_to_cls[key]
    metric_obj = metric_cls(**metric_kwargs)
  else:
    raise ValueError("Unsupported metric: {}".format(key))

  return metric_obj


def default_keras_metrics() -> List[tf.keras.metrics.Metric]:
  """Returns a list of ranking metrics.

  Returns:
    A list of metrics of type `tf.keras.metrics.Metric`.
  """
  list_kwargs = [
      dict(key="ndcg", topn=topn, name="metric/ndcg_{}".format(topn))
      for topn in [1, 3, 5, 10]
  ] + [
      dict(key="arp", name="metric/arp"),
      dict(key="ordered_pair_accuracy", name="metric/ordered_pair_accuracy"),
      dict(key="mrr", name="metric/mrr"),
      dict(key="precision", name="metric/precision"),
      dict(key="map", name="metric/map"),
      dict(key="dcg", name="metric/dcg"),
      dict(key="ndcg", name="metric/ndcg")
  ]
  return [get(**kwargs) for kwargs in list_kwargs]


class _RankingMetric(tf.keras.metrics.Mean):
  """Implements base ranking metric class.

  Please see tf.keras.metrics.Mean for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(self, name=None, dtype=None, **kwargs):
    super(_RankingMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    # An instance of `metrics_impl._RankingMetric`.
    # Overwrite this in subclasses.
    self._metric = None

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)

    # TODO: Add mask argument for metric.compute() call
    per_list_metric_val, per_list_metric_weights = self._metric.compute(
        y_true, y_pred, sample_weight)
    return super(_RankingMetric, self).update_state(
        per_list_metric_val, sample_weight=per_list_metric_weights)


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class MRRMetric(_RankingMetric):
  """Implements mean reciprocal rank (MRR)."""

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(MRRMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.MRRMetric(name=name, topn=topn, ragged=ragged)

  def get_config(self):
    config = super(MRRMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class ARPMetric(_RankingMetric):
  """Implements average relevance position (ARP)."""

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(ARPMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._metric = metrics_impl.ARPMetric(name=name, ragged=ragged)


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class PrecisionMetric(_RankingMetric):
  """Implements precision@k (P@k)."""

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(PrecisionMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.PrecisionMetric(name=name, topn=topn,
                                                ragged=ragged)

  def get_config(self):
    config = super(PrecisionMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


# TODO Add recall metrics to TF1 in another cl.
@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class RecallMetric(_RankingMetric):
  """Implements recall@k."""

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(RecallMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.RecallMetric(name=name, topn=topn,
                                             ragged=ragged)

  def get_config(self):
    config = super(RecallMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class PrecisionIAMetric(_RankingMetric):
  """Implements PrecisionIA@k (Pre-IA@k)."""

  def __init__(self,
               name=None,
               topn=None,
               dtype=None,
               ragged=False,
               **kwargs):
    super(PrecisionIAMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.PrecisionIAMetric(name=name, topn=topn,
                                                  ragged=ragged)

  def get_config(self):
    config = super(PrecisionIAMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class MeanAveragePrecisionMetric(_RankingMetric):
  """Implements mean average precision (MAP)."""

  def __init__(self, name=None, topn=None, dtype=None, ragged=False, **kwargs):
    super(MeanAveragePrecisionMetric, self).__init__(
        name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.MeanAveragePrecisionMetric(name=name, topn=topn,
                                                           ragged=ragged)

  def get_config(self):
    base_config = super(MeanAveragePrecisionMetric, self).get_config()
    config = {
        "topn": self._topn,
    }
    config.update(base_config)
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class NDCGMetric(_RankingMetric):
  """Implements normalized discounted cumulative gain (NDCG).

  The `gain_fn` and `rank_discount_fn` should be keras serializable. Please see
  the `pow_minus_1` and `log2_inverse` above as examples when defining user
  customized functions.
  """

  def __init__(self,
               name=None,
               topn=None,
               gain_fn=None,
               rank_discount_fn=None,
               dtype=None,
               ragged=False,
               **kwargs):
    super(NDCGMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._gain_fn = gain_fn or utils.pow_minus_1
    self._rank_discount_fn = rank_discount_fn or utils.log2_inverse
    self._metric = metrics_impl.NDCGMetric(
        name=name,
        topn=topn,
        gain_fn=self._gain_fn,
        rank_discount_fn=self._rank_discount_fn,
        ragged=ragged)

  def get_config(self):
    base_config = super(NDCGMetric, self).get_config()
    config = {
        "topn": self._topn,
        "gain_fn": self._gain_fn,
        "rank_discount_fn": self._rank_discount_fn,
    }
    config.update(base_config)
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class DCGMetric(_RankingMetric):
  """Implements discounted cumulative gain (DCG).

  The `gain_fn` and `rank_discount_fn` should be keras serializable. Please see
  the `pow_minus_1` and `log2_inverse` above as examples when defining user
  customized functions.
  """

  def __init__(self,
               name=None,
               topn=None,
               gain_fn=None,
               rank_discount_fn=None,
               dtype=None,
               ragged=False,
               **kwargs):
    super(DCGMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._gain_fn = gain_fn or utils.pow_minus_1
    self._rank_discount_fn = rank_discount_fn or utils.log2_inverse
    self._metric = metrics_impl.DCGMetric(
        name=name,
        topn=topn,
        gain_fn=self._gain_fn,
        rank_discount_fn=self._rank_discount_fn,
        ragged=ragged)

  def get_config(self):
    base_config = super(DCGMetric, self).get_config()
    config = {
        "topn": self._topn,
        "gain_fn": self._gain_fn,
        "rank_discount_fn": self._rank_discount_fn,
    }
    config.update(base_config)
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class AlphaDCGMetric(_RankingMetric):
  """Implements alpha discounted cumulative gain (alphaDCG).

  The `rank_discount_fn` should be keras serializable. Please see the
  `log2_inverse` above examples when defining user customized functions.
  """

  def __init__(self,
               name="alpha_dcg_metric",
               topn=None,
               alpha=0.5,
               rank_discount_fn=None,
               seed=None,
               dtype=None,
               ragged=False,
               **kwargs):
    """Construct the ranking metric class for alpha-DCG.

    Args:
      name: A string used as the name for this metric.
      topn: A cutoff for how many examples to consider for this metric.
      alpha: A float between 0 and 1, parameter used in definition of alpha-DCG.
        Introduced as an assessor error in judging whether a document is
        covering a subtopic of the query.
      rank_discount_fn: A function of rank discounts. Default is set to discount
        = 1 / log2(rank+1). The `rank_discount_fn` should be keras serializable.
        Please see the `log2_inverse` above as an example when defining user
        customized functions.
      seed: The ops-level random seed used in shuffle ties in `sort_by_scores`.
      dtype: Data type of the metric output. See `tf.keras.metrics.Metric`.
      ragged: A bool indicating whether the supplied tensors are ragged. If
        True y_true, y_pred and sample_weight (if providing per-example weights)
        need to be ragged tensors with compatible shapes.
      **kwargs: Other keyward arguments used in `tf.keras.metrics.Metric`.
    """
    super(AlphaDCGMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._alpha = alpha
    self._rank_discount_fn = rank_discount_fn or utils.log2_inverse
    self._seed = seed
    self._metric = metrics_impl.AlphaDCGMetric(
        name=name,
        topn=topn,
        alpha=alpha,
        rank_discount_fn=self._rank_discount_fn,
        seed=seed,
        ragged=ragged)

  def get_config(self):
    config = super(AlphaDCGMetric, self).get_config()
    config.update({
        "topn": self._topn,
        "alpha": self._alpha,
        "rank_discount_fn": self._rank_discount_fn,
        "seed": self._seed,
    })
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class OPAMetric(_RankingMetric):
  """Implements ordered pair accuracy (OPA)."""

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(OPAMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._metric = metrics_impl.OPAMetric(name=name, ragged=ragged)
