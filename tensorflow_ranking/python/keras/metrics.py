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
"""Keras metrics in TF-Ranking.

NOTE: For metrics that compute a ranking, ties are broken randomly. This means
that metrics may be stochastic if items with equal scores are provided.

WARNING: Some metrics (e.g. Recall or MRR) are not well-defined when there are
no relevant items (e.g. if `y_true` has a row of only zeroes). For these cases,
the TF-Ranking metrics will evaluate to `0`.
"""

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


def default_keras_metrics(**kwargs) -> List[tf.keras.metrics.Metric]:
  """Returns a list of ranking metrics.

  Args:
    **kwargs: Additional kwargs to pass to each keras metric.

  Returns:
    A list of metrics of type `tf.keras.metrics.Metric`.
  """
  list_kwargs = [
      dict(key="ndcg", topn=topn, name="metric/ndcg_{}".format(topn), **kwargs)
      for topn in [1, 3, 5, 10]
  ] + [
      dict(key="arp", name="metric/arp", **kwargs),
      dict(key="ordered_pair_accuracy", name="metric/ordered_pair_accuracy",
           **kwargs),
      dict(key="mrr", name="metric/mrr", **kwargs),
      dict(key="precision", name="metric/precision", **kwargs),
      dict(key="map", name="metric/map", **kwargs),
      dict(key="dcg", name="metric/dcg", **kwargs),
      dict(key="ndcg", name="metric/ndcg", **kwargs)
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
  r"""Mean reciprocal rank (MRR).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  MRR(y, s) = max_i y_i / rank(s_i)
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> mrr = tfr.keras.metrics.MRRMetric()
  >>> mrr(y_true, y_pred).numpy()
  0.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> mrr = tfr.keras.metrics.MRRMetric(ragged=True)
  >>> mrr(y_true, y_pred).numpy()
  0.75

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.MRRMetric()])
  ```

  Definition:

  $$
  \text{MRR}(\{y\}, \{s\}) = \max_i \frac{\bar{y}_i}{\text{rank}(s_i)}
  $$

  where $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores
  $$s$$ with ties broken randomly and $$\bar{y_i}$$ are truncated labels:

  $$
  \bar{y}_i = \begin{cases}
  1 & \text{if }y_i \geq 1 \\
  0 & \text{else}
  \end{cases}
  $$
  """

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
  r"""Average relevance position (ARP).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  ARP(y, s) = sum_i (y_i * rank(s_i)) / sum_j y_j
  ```

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> arp = tfr.keras.metrics.ARPMetric()
  >>> arp(y_true, y_pred).numpy()
  2.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> arp = tfr.keras.metrics.ARPMetric(ragged=True)
  >>> arp(y_true, y_pred).numpy()
  1.75

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.ARPMetric()])
  ```

  Definition:

  $$
  \text{ARP}(\{y\}, \{s\}) =
  \frac{1}{\sum_i y_i} \sum_i y_i \cdot \text{rank}(s_i)
  $$

  where $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores
  $$s$$ with ties broken randomly.
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(ARPMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._metric = metrics_impl.ARPMetric(name=name, ragged=ragged)


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
class PrecisionMetric(_RankingMetric):
  r"""Precision@k (P@k).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  P@K(y, s) = 1/k sum_i I[rank(s_i) < k] y_i
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> precision_at_2 = tfr.keras.metrics.PrecisionMetric(topn=2)
  >>> precision_at_2(y_true, y_pred).numpy()
  0.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> precision_at_2 = tfr.keras.metrics.PrecisionMetric(topn=2, ragged=True)
  >>> precision_at_2(y_true, y_pred).numpy()
  0.5

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.PrecisionMetric()])
  ```

  Definition:

  $$
  \text{P@k}(\{y\}, \{s\}) =
  \frac{1}{k} \sum_i I[\text{rank}(s_i) \leq k] \bar{y}_i
  $$

  where:

  * $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores $$s$$
    with ties broken randomly
  * $$I[]$$ is the indicator function:\
    $$I[\text{cond}] = \begin{cases}
    1 & \text{if cond is true}\\
    0 & \text{else}\end{cases}
    $$
  * $$\bar{y}_i$$ are the truncated labels:\
    $$
    \bar{y}_i = \begin{cases}
    1 & \text{if }y_i \geq 1 \\
    0 & \text{else}
    \end{cases}
    $$
  * $$k = |y|$$ if $$k$$ is not provided
  """

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
  r"""Recall@k (R@k).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  R@K(y, s) = sum_i I[rank(s_i) < k] y_i / sum_j y_j
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> recall_at_2 = tfr.keras.metrics.RecallMetric(topn=2)
  >>> recall_at_2(y_true, y_pred).numpy()
  0.5

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> recall_at_2 = tfr.keras.metrics.RecallMetric(topn=2, ragged=True)
  >>> recall_at_2(y_true, y_pred).numpy()
  0.75

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.RecallMetric()])
  ```

  Definition:

  $$
  \text{R@k}(\{y\}, \{s\}) =
  \frac{\sum_i I[\text{rank}(s_i) \leq k] \bar{y}_i}{\sum_j \bar{y}_j}
  $$

  where:

  * $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores $$s$$
    with ties broken randomly
  * $$I[]$$ is the indicator function:\
    $$I[\text{cond}] = \begin{cases}
    1 & \text{if cond is true}\\
    0 & \text{else}\end{cases}
    $$
  * $$\bar{y}_i$$ are the truncated labels:\
    $$
    \bar{y}_i = \begin{cases}
    1 & \text{if }y_i \geq 1 \\
    0 & \text{else}
    \end{cases}
    $$
  * $$k = |y|$$ if $$k$$ is not provided
  """

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
  r"""Precision-IA@k (Pre-IA@k).

  Intent-aware Precision@k ([Agrawal et al, 2009][agrawal2009];
  [Clarke et al, 2009][clarke2009]) is a precision metric that operates on
  subtopics and is typically used for diversification tasks..

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  Pre-IA@k(y, s) = sum_t sum_i I[rank(s_i) <= k] y_{i,t} / (# of subtopics * k)
  ```

  NOTE: The labels `y_true` should be of shape
  `[batch_size, list_size, subtopic_size]`, indicating relevance for each
  subtopic in the last dimension.

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_{i,t} = 1` if `y_{i,t} >= 1`.

  Standalone usage:

  >>> y_true = [[[0., 1.], [1., 0.], [1., 1.]]]
  >>> y_pred = [[3., 1., 2.]]
  >>> pre_ia = tfr.keras.metrics.PrecisionIAMetric()
  >>> pre_ia(y_true, y_pred).numpy()
  0.6666667

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant(
  ...   [[[0., 0.], [1., 0.]], [[1., 1.], [0., 2.], [1., 0.]]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> pre_ia = tfr.keras.metrics.PrecisionIAMetric(ragged=True)
  >>> pre_ia(y_true, y_pred).numpy()
  0.5833334

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                metrics=[tfr.keras.metrics.PrecisionIAMetric()])
  ```

  Definition:

  $$
  \text{Pre-IA@k}(y, s) = \frac{1}{\text{# of subtopics} \cdot k}
  \sum_t \sum_i I[\text{rank}(s_i) \leq k] y_{i,t}
  $$

  where $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores
  $$s$$ with ties broken randomly.

  References:

    - [Diversifying Search Results, Agrawal et al, 2009][agrawal2009]
    - [Overview of the TREC 2009 Web Track, Clarke et al, 2009][clarke2009]

  [agrawal2009]:
  https://www.microsoft.com/en-us/research/publication/diversifying-search-results/
  [clarke2009]: https://trec.nist.gov/pubs/trec18/papers/ENT09.OVERVIEW.pdf
  """

  def __init__(self,
               name=None,
               topn=None,
               dtype=None,
               ragged=False,
               **kwargs):
    """Constructor.

    Args:
      name: A string used as the name for this metric.
      topn: A cutoff for how many examples to consider for this metric.
      dtype: Data type of the metric output. See `tf.keras.metrics.Metric`.
      ragged: A bool indicating whether the supplied tensors are ragged. If
        True y_true, y_pred and sample_weight (if providing per-example weights)
        need to be ragged tensors with compatible shapes.
      **kwargs: Other keyward arguments used in `tf.keras.metrics.Metric`.
    """
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
  r"""Mean average precision (MAP).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  MAP(y, s) = sum_k (P@k(y, s) * rel(k)) / sum_i y_i
  rel(k) = y_i if rank(s_i) = k
  ```

  NOTE: This metric converts graded relevance to binary relevance by setting
  `y_i = 1` if `y_i >= 1`.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> map_metric = tfr.keras.metrics.MeanAveragePrecisionMetric(topn=2)
  >>> map_metric(y_true, y_pred).numpy()
  0.25

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> map_metric = tfr.keras.metrics.MeanAveragePrecisionMetric(
  ...   topn=2, ragged=True)
  >>> map_metric(y_true, y_pred).numpy()
  0.5

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                metrics=[tfr.keras.metrics.MeanAveragePrecisionMetric()])
  ```

  Definition:

  $$
  \text{MAP}(\{y\}, \{s\}) =
  \frac{\sum_k P@k(y, s) \cdot \text{rel}(k)}{\sum_j \bar{y}_j} \\
  \text{rel}(k) = \max_i I[\text{rank}(s_i) = k] \bar{y}_i
  $$

  where:

  * $$P@k(y, s)$$ is the Precision at rank $$k$$. See
    `tfr.keras.metrics.PrecisionMetric`.
  * $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores $$s$$
    with ties broken randomly
  * $$I[]$$ is the indicator function:\
    $$I[\text{cond}] = \begin{cases}
    1 & \text{if cond is true}\\
    0 & \text{else}\end{cases}
    $$
  * $$\bar{y}_i$$ are the truncated labels:\
    $$
    \bar{y}_i = \begin{cases}
    1 & \text{if }y_i \geq 1 \\
    0 & \text{else}
    \end{cases}
    $$
  """

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
  r"""Normalized discounted cumulative gain (NDCG).

  Normalized discounted cumulative gain ([Järvelin et al, 2002][jarvelin2002])
  is the normalized version of `tfr.keras.metrics.DCGMetric`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  NDCG(y, s) = DCG(y, s) / DCG(y, y)
  DCG(y, s) = sum_i gain(y_i) * rank_discount(rank(s_i))
  ```

  NOTE: The `gain_fn` and `rank_discount_fn` should be keras serializable.
  Please see `tfr.keras.utils.pow_minus_1` and `tfr.keras.utils.log2_inverse` as
  examples when defining user customized functions.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> ndcg = tfr.keras.metrics.NDCGMetric()
  >>> ndcg(y_true, y_pred).numpy()
  0.6934264

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> ndcg = tfr.keras.metrics.NDCGMetric(ragged=True)
  >>> ndcg(y_true, y_pred).numpy()
  0.7974351

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.NDCGMetric()])
  ```

  Definition:

  $$
  \text{NDCG}(\{y\}, \{s\}) =
  \frac{\text{DCG}(\{y\}, \{s\})}{\text{DCG}(\{y\}, \{y\})} \\
  \text{DCG}(\{y\}, \{s\}) =
  \sum_i \text{gain}(y_i) \cdot \text{rank_discount}(\text{rank}(s_i))
  $$

  where $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores
  $$s$$ with ties broken randomly.

  References:

    - [Cumulated gain-based evaluation of IR techniques, Järvelin et al,
       2002][jarvelin2002]

  [jarvelin2002]: https://dl.acm.org/doi/10.1145/582415.582418
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
  r"""Discounted cumulative gain (DCG).

  Discounted cumulative gain ([Järvelin et al, 2002][jarvelin2002]).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  DCG(y, s) = sum_i gain(y_i) * rank_discount(rank(s_i))
  ```

  NOTE: The `gain_fn` and `rank_discount_fn` should be keras serializable.
  Please see `tfr.keras.utils.pow_minus_1` and `tfr.keras.utils.log2_inverse` as
  examples when defining user customized functions.

  Standalone usage:

  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> dcg = tfr.keras.metrics.DCGMetric()
  >>> dcg(y_true, y_pred).numpy()
  1.1309297

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> dcg = tfr.keras.metrics.DCGMetric(ragged=True)
  >>> dcg(y_true, y_pred).numpy()
  2.065465

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.DCGMetric()])
  ```

  Definition:

  $$
  \text{DCG}(\{y\}, \{s\}) =
  \sum_i \text{gain}(y_i) \cdot \text{rank_discount}(\text{rank}(s_i))
  $$

  where $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores
  $$s$$ with ties broken randomly.

  References:

    - [Cumulated gain-based evaluation of IR techniques, Järvelin et al,
       2002][jarvelin2002]

  [jarvelin2002]: https://dl.acm.org/doi/10.1145/582415.582418
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
  r"""Alpha discounted cumulative gain (alphaDCG).

  Alpha discounted cumulative gain ([Clarke et al, 2008][clarke2008];
  [Clarke et al, 2009][clarke2009]) is a cumulative gain metric that operates
  on subtopics and is typically used for diversification tasks.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  alphaDCG(y, s) = sum_t sum_i gain(y_{i,t}) * rank_discount(rank(s_i))
  gain(y_{i,t}) = (1 - alpha)^(sum_j I[rank(s_j) < rank(s_i)] * gain(y_{j,t}))
  ```

  NOTE: The labels `y_true` should be of shape
  `[batch_size, list_size, subtopic_size]`, indicating relevance for each
  subtopic in the last dimension.

  NOTE: The `rank_discount_fn` should be keras serializable. Please see
  `tfr.keras.utils.log2_inverse` as an example when defining user customized
  functions.

  Standalone usage:

  >>> y_true = [[[0., 1.], [1., 0.], [1., 1.]]]
  >>> y_pred = [[3., 1., 2.]]
  >>> alpha_dcg = tfr.keras.metrics.AlphaDCGMetric()
  >>> alpha_dcg(y_true, y_pred).numpy()
  2.1963947

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant(
  ...   [[[0., 0.], [1., 0.]], [[1., 1.], [0., 2.], [1., 0.]]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> alpha_dcg = tfr.keras.metrics.AlphaDCGMetric(ragged=True)
  >>> alpha_dcg(y_true, y_pred).numpy()
  1.8184297

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.AlphaDCGMetric()])
  ```

  Definition:

  $$
  \alpha\text{DCG}(y, s) =
  \sum_t \sum_i \text{gain}(y_{i, t}, \alpha)
  \text{ rank_discount}(\text{rank}(s_i))\\
  \text{gain}(y_{i, t}, \alpha) =
  y_{i, t} (1 - \alpha)^{\sum_j I[\text{rank}(s_j) < \text{rank}(s_i)] y_{j, t}}
  $$

  where $$\text{rank}(s_i)$$ is the rank of item $$i$$ after sorting by scores
  $$s$$ with ties broken randomly and $$I[]$$ is the indicator function:

  $$
  I[\text{cond}] = \begin{cases}
  1 & \text{if cond is true}\\
  0 & \text{else}\end{cases}
  $$

  References:

    - [Novelty and diversity in information retrieval evaluation, Clarke et al,
       2008][clarke2008]
    - [Overview of the TREC 2009 Web Track, Clarke et al, 2009][clarke2009]

  [clarke2008]: https://dl.acm.org/doi/10.1145/1390334.1390446
  [clarke2009]: https://trec.nist.gov/pubs/trec18/papers/ENT09.OVERVIEW.pdf
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
      rank_discount_fn: A function of rank discounts. Default is set to
        `1 / log2(rank+1)`. The `rank_discount_fn` should be keras serializable.
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
  r"""Ordered pair accuracy (OPA).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  OPA(y, s) = sum_i sum_j I[s_i > s_j] I[y_i > y_j] / sum_i sum_j I[y_i > y_j]
  ```

  NOTE: Pairs with equal labels (`y_i = y_j`) are always ignored. Pairs with
  equal scores (`s_i = s_j`) are considered incorrectly ordered.

  Standalone usage:

  >>> y_true = [[0., 1., 2.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> opa = tfr.keras.metrics.OPAMetric()
  >>> opa(y_true, y_pred).numpy()
  0.33333334

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> opa = tfr.keras.metrics.OPAMetric(ragged=True)
  >>> opa(y_true, y_pred).numpy()
  0.5

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.OPAMetric()])
  ```

  Definition:

  $$
  \text{OPA}(\{y\}, \{s\}) =
  \frac{\sum_i \sum_j I[s_i > s_j] I[y_i > y_j]}{\sum_i \sum_j I[y_i > y_j]}
  $$

  where $$I[]$$ is the indicator function:

  $$
  I[\text{cond}] = \begin{cases}
  1 & \text{if cond is true}\\
  0 & \text{else}\end{cases}
  $$
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(OPAMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    self._metric = metrics_impl.OPAMetric(name=name, ragged=ragged)
