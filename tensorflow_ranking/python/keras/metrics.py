# Copyright 2020 The TensorFlow Ranking Authors.
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

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python import metrics_impl

_DEFAULT_GAIN_FN = lambda label: tf.pow(2.0, label) - 1

_DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)


def default_keras_metrics():
  """Returns a list of ranking metrics.

  Returns:
    A list of metrics of type `tf.keras.metrics.Metric`.
  """
  metrics = [
      NDCGMetric(name="metric/ndcg_{}".format(topn), topn=topn)
      for topn in [1, 3, 5, 10]
  ]
  metrics.extend([
      ARPMetric(name="metric/arp"),
      OPAMetric(name="metric/ordered_pair_accuracy"),
      MRRMetric(name="metric/mrr"),
      PrecisionMetric(name="metric/precision"),
      MeanAveragePrecisionMetric(name="metric/map"),
      DCGMetric(name="metric/dcg"),
      NDCGMetric(name="metric/ndcg")
  ])
  return metrics


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

    per_list_metric_val, per_list_metric_weights = self._metric.compute(
        y_true, y_pred, sample_weight)
    return super(_RankingMetric, self).update_state(
        per_list_metric_val, sample_weight=per_list_metric_weights)


class MRRMetric(_RankingMetric):
  """Implements mean reciprocal rank (MRR)."""

  def __init__(self, name=None, topn=None, dtype=None, **kwargs):
    super(MRRMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.MRRMetric(name=name, topn=topn)

  def get_config(self):
    config = super(MRRMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


class ARPMetric(_RankingMetric):
  """Implements average relevance position (ARP)."""

  def __init__(self, name=None, dtype=None, **kwargs):
    super(ARPMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._metric = metrics_impl.ARPMetric(name=name)


class PrecisionMetric(_RankingMetric):
  """Implements precision@k (P@k)."""

  def __init__(self, name=None, topn=None, dtype=None, **kwargs):
    super(PrecisionMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.PrecisionMetric(name=name, topn=topn)

  def get_config(self):
    config = super(PrecisionMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


class PrecisionIAMetric(_RankingMetric):
  """Implements PrecisionIA@k (Pre-IA@k)."""

  def __init__(self,
               topn=None,
               dtype=None,
               name="precision_ia_metric",
               **kwargs):
    super(PrecisionIAMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.PrecisionIAMetric(name=name, topn=topn)

  def get_config(self):
    config = super(PrecisionIAMetric, self).get_config()
    config.update({
        "topn": self._topn,
    })
    return config


class MeanAveragePrecisionMetric(_RankingMetric):
  """Implements mean average precision (MAP)."""

  def __init__(self, name=None, topn=None, dtype=None, **kwargs):
    super(MeanAveragePrecisionMetric, self).__init__(
        name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._metric = metrics_impl.MeanAveragePrecisionMetric(name=name, topn=topn)

  def get_config(self):
    base_config = super(MeanAveragePrecisionMetric, self).get_config()
    config = {
        "topn": self._topn,
    }
    config.update(base_config)
    return config


class NDCGMetric(_RankingMetric):
  """Implements normalized discounted cumulative gain (NDCG)."""

  def __init__(self,
               name=None,
               topn=None,
               gain_fn=_DEFAULT_GAIN_FN,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
               dtype=None,
               **kwargs):
    super(NDCGMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._metric = metrics_impl.NDCGMetric(
        name=name,
        topn=topn,
        gain_fn=gain_fn,
        rank_discount_fn=rank_discount_fn)

  def get_config(self):
    base_config = super(NDCGMetric, self).get_config()
    config = {
        "topn": self._topn,
        "gain_fn": self._gain_fn,
        "rank_discount_fn": self._rank_discount_fn
    }
    config.update(base_config)
    return config


class DCGMetric(_RankingMetric):
  """Implements discounted cumulative gain (DCG)."""

  def __init__(self,
               name=None,
               topn=None,
               gain_fn=_DEFAULT_GAIN_FN,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
               dtype=None,
               **kwargs):
    super(DCGMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._metric = metrics_impl.DCGMetric(
        name=name,
        topn=topn,
        gain_fn=gain_fn,
        rank_discount_fn=rank_discount_fn)

  def get_config(self):
    config = super(DCGMetric, self).get_config()
    config.update({
        "topn": self._topn,
        "gain_fn": self._gain_fn,
        "rank_discount_fn": self._rank_discount_fn
    })
    return config


class AlphaDCGMetric(_RankingMetric):
  """Implements alpha discounted cumulative gain (alphaDCG)."""

  def __init__(self,
               topn=None,
               alpha=0.5,
               rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
               seed=None,
               dtype=None,
               name="alpha_dcg_metric",
               **kwargs):
    """Construct the ranking metric class for alpha-DCG.

    Args:
      topn: A cutoff for how many examples to consider for this metric.
      alpha: A float between 0 and 1, parameter used in definition of alpha-DCG.
        Introduced as an assessor error in judging whether a document is
        covering a subtopic of the query.
      rank_discount_fn: A function of rank discounts. Default is set to
        discount = 1 / log2(rank+1).
      seed: The ops-level random seed used in shuffle ties in `sort_by_scores`.
      dtype: Data type of the metric output. See `tf.keras.metrics.Metric`.
      name: A string used as the name for this metric.
      **kwargs: Other keyward arguments used in `tf.keras.metrics.Metric`.
    """
    super(AlphaDCGMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._topn = topn
    self._alpha = alpha
    self._rank_discount_fn = rank_discount_fn
    self._seed = seed
    self._metric = metrics_impl.AlphaDCGMetric(
        name=name,
        topn=topn,
        alpha=alpha,
        rank_discount_fn=rank_discount_fn,
        seed=seed)

  def get_config(self):
    config = super(AlphaDCGMetric, self).get_config()
    config.update({
        "topn": self._topn,
        "alpha": self._alpha,
        "rank_discount_fn": self._rank_discount_fn,
        "seed": self._seed,
    })
    return config


class OPAMetric(_RankingMetric):
  """Implements ordered pair accuracy (OPA)."""

  def __init__(self, name=None, dtype=None, **kwargs):
    super(OPAMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._metric = metrics_impl.OPAMetric(name=name)
