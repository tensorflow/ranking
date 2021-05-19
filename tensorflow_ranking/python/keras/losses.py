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
"""Keras losses in TF-Ranking."""

import tensorflow.compat.v2 as tf

from tensorflow.python.keras.utils import losses_utils
from tensorflow_ranking.python import losses_impl
from tensorflow_ranking.python.keras import utils


class RankingLossKey(object):
  """Ranking loss key strings."""
  # Names for the ranking based loss functions.
  PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
  PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
  PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
  SOFTMAX_LOSS = 'softmax_loss'
  UNIQUE_SOFTMAX_LOSS = 'unique_softmax_loss'
  SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
  MEAN_SQUARED_LOSS = 'mean_squared_loss'
  LIST_MLE_LOSS = 'list_mle_loss'
  APPROX_NDCG_LOSS = 'approx_ndcg_loss'
  APPROX_MRR_LOSS = 'approx_mrr_loss'
  GUMBEL_APPROX_NDCG_LOSS = 'gumbel_approx_ndcg_loss'
  # TODO: Add support for circle loss and neural sort losses.


def get(loss,
        reduction=tf.losses.Reduction.AUTO,
        lambda_weight=None,
        name=None,
        **kwargs):
  """Factory method to get a ranking loss class.

  Args:
    loss: (str) An attribute of `RankingLossKey`, defining which loss object to
      return.
    reduction: (enum)  An enum of strings indicating the loss reduction type.
      See type definition in the `tf.compat.v2.losses.Reduction`.
    lambda_weight: (losses_impl._LambdaWeight) A lambda object for ranking
      metric optimization.
    name: (optional) (str) Name of loss.
    **kwargs: Keyword arguments for the loss object.

  Returns:
    A ranking loss instance. See `_RankingLoss` signature for more details.

  Raises:
    ValueError: If loss_key is unsupported.
  """
  loss_kwargs = {'reduction': reduction, 'name': name}
  if kwargs:
    loss_kwargs.update(kwargs)

  loss_kwargs_with_lambda_weights = {'lambda_weight': lambda_weight}
  loss_kwargs_with_lambda_weights.update(loss_kwargs)

  key_to_cls = {
      RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS: SigmoidCrossEntropyLoss,
      RankingLossKey.MEAN_SQUARED_LOSS: MeanSquaredLoss,
      RankingLossKey.APPROX_NDCG_LOSS: ApproxNDCGLoss,
      RankingLossKey.APPROX_MRR_LOSS: ApproxMRRLoss,
      RankingLossKey.GUMBEL_APPROX_NDCG_LOSS: GumbelApproxNDCGLoss,
  }

  key_to_cls_with_lambda_weights = {
      RankingLossKey.LIST_MLE_LOSS: ListMLELoss,
      RankingLossKey.PAIRWISE_HINGE_LOSS: PairwiseHingeLoss,
      RankingLossKey.PAIRWISE_LOGISTIC_LOSS: PairwiseLogisticLoss,
      RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS: PairwiseSoftZeroOneLoss,
      RankingLossKey.SOFTMAX_LOSS: SoftmaxLoss,
      RankingLossKey.UNIQUE_SOFTMAX_LOSS: UniqueSoftmaxLoss,
  }
  if loss in key_to_cls:
    loss_cls = key_to_cls[loss]
    loss_obj = loss_cls(**loss_kwargs)
  elif loss in key_to_cls_with_lambda_weights:
    loss_cls = key_to_cls_with_lambda_weights[loss]
    loss_obj = loss_cls(**loss_kwargs_with_lambda_weights)
  else:
    raise ValueError('unsupported loss: {}'.format(loss))

  return loss_obj


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class DCGLambdaWeight(losses_impl.DCGLambdaWeight):
  """Keras serializable class for DCG."""

  def __init__(self,
               topn=None,
               gain_fn=None,
               rank_discount_fn=None,
               normalized=False,
               smooth_fraction=0.,
               **kwargs):
    gain_fn = gain_fn or utils.identity
    rank_discount_fn = rank_discount_fn or utils.inverse
    super().__init__(topn, gain_fn, rank_discount_fn, normalized,
                     smooth_fraction)

  def get_config(self):
    return {
        'topn': self._topn,
        'gain_fn': self._gain_fn,
        'rank_discount_fn': self._rank_discount_fn,
        'normalized': self._normalized,
        'smooth_fraction': self._smooth_fraction,
    }


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class NDCGLambdaWeight(DCGLambdaWeight):
  """Keras serializable class for NDCG."""

  def __init__(self,
               topn=None,
               gain_fn=None,
               rank_discount_fn=None,
               smooth_fraction=0.,
               **kwargs):
    super().__init__(
        topn,
        gain_fn or utils.pow_minus_1,
        rank_discount_fn or utils.log2_inverse,
        normalized=True,
        smooth_fraction=smooth_fraction)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class PrecisionLambdaWeight(losses_impl.PrecisionLambdaWeight):
  """Keras serializable class for Precision."""

  def __init__(self, topn=None, positive_fn=None, **kwargs):
    positive_fn = positive_fn or utils.is_greater_equal_1
    super().__init__(topn, positive_fn)

  def get_config(self):
    return {
        'topn': self._topn,
        'positive_fn': self._positive_fn,
    }


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ListMLELambdaWeight(losses_impl.ListMLELambdaWeight):

  def __init__(self, rank_discount_fn, **kwargs):
    super().__init__(rank_discount_fn)

  def get_config(self):
    return {
        'rank_discount_fn': self._rank_discount_fn,
    }


class _RankingLoss(tf.keras.losses.Loss):
  """Base class for all ranking losses.

  Please see tf.keras.losses.Loss for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(self, reduction=tf.losses.Reduction.AUTO, name=None,
               ragged=False):
    super().__init__(reduction, name)
    # An instance of loss in `losses_impl`. Overwrite this in subclasses.
    self._loss = None
    self._ragged = ragged

  def __call__(self, y_true, y_pred, sample_weight=None):
    """See tf.keras.losses.Loss."""
    if self._loss is None:
      raise ValueError('self._loss is not defined. Please use a subclass.')
    sample_weight = self._loss.normalize_weights(y_true, sample_weight)
    return super().__call__(y_true, y_pred, sample_weight)

  def call(self, y_true, y_pred):
    """See tf.keras.losses.Loss."""
    y_pred = self._loss.get_logits(y_pred)
    losses, weights = self._loss.compute_unreduced_loss(
        labels=y_true, logits=y_pred)
    return tf.multiply(losses, weights)

  def get_config(self):
    config = super().get_config()
    config.update({
        'ragged': self._ragged
    })
    return config


class _PairwiseLoss(_RankingLoss):
  """Base class for pairwise ranking losses."""

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False,
               **kwargs):
    super().__init__(reduction, name, ragged)
    self._lambda_weight = lambda_weight
    self._temperature = temperature

  def get_config(self):
    config = super().get_config()
    config.update({
        'lambda_weight':
            tf.keras.utils.serialize_keras_object(self._lambda_weight),
        'temperature':
            self._temperature,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    config.update({
        'lambda_weight':
            tf.keras.utils.deserialize_keras_object(config['lambda_weight']),
    })
    return cls(**config)

  def call(self, y_true, y_pred):
    """See _RankingLoss."""
    losses, weights = self._loss.compute_unreduced_loss(
        labels=y_true, logits=y_pred)
    losses = tf.multiply(losses, weights)
    # [batch_size, list_size, list_size]
    losses.get_shape().assert_has_rank(3)
    # Reduce the loss along the last dim so that weights ([batch_size, 1] or
    # [batch_size, list_size] can be applied in __call__.
    return tf.reduce_sum(losses, axis=2)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class PairwiseHingeLoss(_PairwiseLoss):
  r"""Computes pairwise hinge loss between `y_true` and `y_pred`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = sum_i sum_j I[y_i > y_j] * max(0, 1 - (s_i - s_j))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.PairwiseHingeLoss()
  >>> loss(y_true, y_pred).numpy()
  0.6

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.PairwiseHingeLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.41666666

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.PairwiseHingeLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i \sum_j I[y_i > y_j] \max(0, 1 - (s_i - s_j))
  $$
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False):
    """Pairwise hinge loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`, or,
        `tfr.keras.losses.PrecisionLambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.PairwiseHingeLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class PairwiseLogisticLoss(_PairwiseLoss):
  r"""Computes pairwise logistic loss between `y_true` and `y_pred`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = sum_i sum_j I[y_i > y_j] * log(1 + exp(-(s_i - s_j)))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.PairwiseLogisticLoss()
  >>> loss(y_true, y_pred).numpy()
  0.39906943

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.PairwiseLogisticLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.3109182

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.PairwiseLogisticLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))
  $$
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False):
    """Pairwise logistic loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`, or,
        `tfr.keras.losses.PrecisionLambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.PairwiseLogisticLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class PairwiseSoftZeroOneLoss(_PairwiseLoss):
  r"""Computes pairwise soft zero-one loss between `y_true` and `y_pred`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = sum_i sum_j I[y_i > y_j] * (1 - sigmoid(s_i - s_j))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.PairwiseSoftZeroOneLoss()
  >>> loss(y_true, y_pred).numpy()
  0.274917

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.PairwiseSoftZeroOneLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.22945064

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tfr.keras.losses.PairwiseSoftZeroOneLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_i \sum_j I[y_i > y_j] (1 - \text{sigmoid}(s_i - s_j))
  $$
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False):
    """Pairwise soft zero one loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`, or,
        `tfr.keras.losses.PrecisionLambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.PairwiseSoftZeroOneLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


class _ListwiseLoss(_RankingLoss):
  """Base class for listwise ranking losses."""

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False,
               **kwargs):
    super().__init__(reduction, name, ragged)
    self._lambda_weight = lambda_weight
    self._temperature = temperature

  def get_config(self):
    config = super().get_config()
    config.update({
        'lambda_weight':
            tf.keras.utils.serialize_keras_object(self._lambda_weight),
        'temperature':
            self._temperature,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    config.update({
        'lambda_weight':
            tf.keras.utils.deserialize_keras_object(config['lambda_weight']),
    })
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class SoftmaxLoss(_ListwiseLoss):
  r"""Computes Softmax cross-entropy loss between `y_true` and `y_pred`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = - sum_i y_i * log(softmax(s_i))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.SoftmaxLoss()
  >>> loss(y_true, y_pred).numpy()
  0.7981389

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.SoftmaxLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.83911896

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.SoftmaxLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) =
  - \sum_i y_i \cdot \log\left(\frac{exp(s_i)}{\sum_j exp(s_j)}\right)
  $$
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False):
    """Softmax cross-entropy loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`, or,
        `tfr.keras.losses.PrecisionLambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.SoftmaxLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)

  def __call__(self, y_true, y_pred, sample_weight=None):
    """See _RankingLoss."""
    losses, sample_weight = self._loss.compute_per_list(
        y_true, y_pred, sample_weight)
    return losses_utils.compute_weighted_loss(
        losses, sample_weight, reduction=self._get_reduction())


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class UniqueSoftmaxLoss(_ListwiseLoss):
  r"""Computes unique softmax cross-entropy loss between `y_true` and `y_pred`.

  Implements unique rating softmax loss ([Zhu et al, 2020][zhu2020]).

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = - sum_i (2^{y_i} - 1) *
                 log(exp(s_i) / sum_j I(y_i > y_j) exp(s_j) + exp(s_i))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.UniqueSoftmaxLoss()
  >>> loss(y_true, y_pred).numpy()
  0.7981389

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.UniqueSoftmaxLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.83911896

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.UniqueSoftmaxLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) =
  - \sum_i (2^{y_i} - 1) \cdot
  \log\left(\frac{\exp(s_i)}{\sum_j I_{y_i > y_j} \exp(s_j) + \exp(s_i)}\right)
  $$

  References:
    - [Listwise Learning to Rank by Exploring Unique Ratings, Zhu et al,
       2020][zhu2020]

  [zhu2020]: https://arxiv.org/abs/2001.01828
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False):
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.UniqueSoftmaxLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ListMLELoss(_ListwiseLoss):
  r"""Computes ListMLE loss between `y_true` and `y_pred`.

  Implements ListMLE loss ([Xia et al, 2008][xia2008]). For each list of scores
  `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = - log P(permutation_y | s)
  P(permutation_y | s) = Plackett-Luce probability of permutation_y given s
  permutation_y = permutation of items sorted by labels y.
  ```

  NOTE: This loss is stochastic and may return different values for identical
  inputs.

  Standalone usage:

  >>> tf.random.set_seed(42)
  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.ListMLELoss()
  >>> loss(y_true, y_pred).numpy()
  0.7981389

  >>> # Using ragged tensors
  >>> tf.random.set_seed(42)
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.ListMLELoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  1.1613163

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.ListMLELoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = - \log(P(\pi_y | s))
  $$

  where $$P(\pi_y | s)$$ is the Plackett-Luce probability of a permutation
  $$\pi_y$$ conditioned on scores $$s$$. Here $$\pi_y$$ represents a permutation
  of items ordered by the relevance labels $$y$$ where ties are broken randomly.

  References:
    - [Listwise approach to learning to rank: theory and algorithm, Xia et al,
       2008][xia2008]

  [xia2008]: https://dl.acm.org/doi/10.1145/1390156.1390306
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=1.0,
               ragged=False):
    """ListMLE loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`,
        `tfr.keras.losses.PrecisionLambdaWeight`, or,
        `tfr.keras.losses.ListMLELambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.ListMLELoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ApproxMRRLoss(_ListwiseLoss):
  r"""Computes approximate MRR loss between `y_true` and `y_pred`.

  Implementation of ApproxMRR loss ([Qin et al, 2008][qin2008]). This loss is
  an approximation for `tfr.keras.metrics.MRRMetric`. It replaces the
  non-differentiable ranking function in MRR with a differentiable approximation
  based on the logistic function.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = sum_i (1 / approxrank(s_i)) * y_i
  approxrank(s_i) = 1 + sum_j (1 / (1 + exp(-(s_j - s_i) / temperature)))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.ApproxMRRLoss()
  >>> loss(y_true, y_pred).numpy()
  -0.53168947

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.ApproxMRRLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  -0.73514676

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.ApproxMRRLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = -\sum_{i} \frac{1}{\text{approxrank}_i} y_i
  $$

  where:

  $$
  \text{approxrank}_i = 1 + \sum_{j \neq i}
  \frac{1}{1 + \exp\left(\frac{-(s_j - s_i)}{\text{temperature}}\right)}
  $$

  References:
    - [A General Approximation Framework for Direct Optimization of Information
       Retrieval Measures, Qin et al, 2008][qin2008]

  [qin2008]: https://www.microsoft.com/en-us/research/publication/a-general-approximation-framework-for-direct-optimization-of-information-retrieval-measures/
  """  # pylint: disable=g-line-too-long

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=0.1,
               ragged=False):
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.ApproxMRRLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ApproxNDCGLoss(_ListwiseLoss):
  r"""Computes approximate NDCG loss between `y_true` and `y_pred`.

  Implementation of ApproxNDCG loss ([Qin et al, 2008][qin2008];
  [Bruch et al, 2019][bruch2019]). This loss is an approximation for
  `tfr.keras.metrics.NDCGMetric`. It replaces the non-differentiable ranking
  function in NDCG with a differentiable approximation based on the logistic
  function.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = sum_i (2^y_i - 1) / log_2(1 + approxrank(s_i))
  approxrank(s_i) = 1 + sum_j (1 / (1 + exp(-(s_j - s_i) / temperature)))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.ApproxNDCGLoss()
  >>> loss(y_true, y_pred).numpy()
  -0.6536734

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.ApproxNDCGLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  -0.80536866

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.ApproxNDCGLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = - \frac{1}{\text{DCG}(y, y)} \sum_{i}
  \frac{2^{y_i} - 1}{\log_2(1 + \text{rank}_i)}
  $$

  where:

  $$
  \text{rank}_i = 1 + \sum_{j \neq i}
  \frac{1}{1 + \exp\left(\frac{-(s_j - s_i)}{\text{temperature}}\right)}
  $$

  References:
    - [A General Approximation Framework for Direct Optimization of Information
       Retrieval Measures, Qin et al, 2008][qin2008]
    - [Revisiting Approximate Metric Optimization in the Age of Deep Neural
       Networks, Bruch et al, 2019][bruch2019]

  [qin2008]:
  https://www.microsoft.com/en-us/research/publication/a-general-approximation-framework-for-direct-optimization-of-information-retrieval-measures/
  [bruch2019]: https://research.google/pubs/pub48168/
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=0.1,
               ragged=False):
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._loss = losses_impl.ApproxNDCGLoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class GumbelApproxNDCGLoss(ApproxNDCGLoss):
  r"""Computes the Gumbel approximate NDCG loss between `y_true` and `y_pred`.

  Implementation of Gumbel ApproxNDCG loss ([Bruch et al, 2020][bruch2020]).
  This loss is the same as `tfr.keras.losses.ApproxNDCGLoss` but where logits
  are sampled from the Gumbel distribution:

  `y_new_pred ~ Gumbel(y_pred, 1 / temperature)`

  NOTE: This loss is stochastic and may return different values for identical
  inputs.

  Standalone usage:

  >>> tf.random.set_seed(42)
  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.GumbelApproxNDCGLoss(seed=42)
  >>> loss(y_true, y_pred).numpy()
  -0.8160851

  >>> # Using a higher gumbel temperature
  >>> loss = tfr.keras.losses.GumbelApproxNDCGLoss(gumbel_temperature=2.0,
  ...     seed=42)
  >>> loss(y_true, y_pred).numpy()
  -0.7583889

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.GumbelApproxNDCGLoss(seed=42, ragged=True)
  >>> loss(y_true, y_pred).numpy()
  -0.69871885

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.GumbelApproxNDCGLoss())
  ```

  Definition:

  $$\mathcal{L}(\{y\}, \{s\}) = \text{ApproxNDCGLoss}(\{y\}, \{z\})$$

  where

  $$
  z \sim \text{Gumbel}(s, \beta)\\
  p(z) = e^{-t-e^{-t}}\\
  t = \beta(z - s)\\
  \beta = \frac{1}{\text{temperature}}
  $$

  References:
    - [A Stochastic Treatment of Learning to Rank Scoring Functions, Bruch et
       al, 2020][bruch2020]

  [bruch2020]: https://research.google/pubs/pub48689/
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               lambda_weight=None,
               temperature=0.1,
               sample_size=8,
               gumbel_temperature=1.0,
               seed=None,
               ragged=False):
    super().__init__(reduction, name, lambda_weight, temperature=temperature,
                     ragged=ragged)
    self._sample_size = sample_size
    self._gumbel_temperature = gumbel_temperature
    self._seed = seed

    self._gumbel_sampler = losses_impl.GumbelSampler(
        name=name,
        sample_size=sample_size,
        temperature=gumbel_temperature,
        seed=seed,
        ragged=ragged)

  def get_config(self):
    config = super().get_config()
    config.update({
        'sample_size': self._sample_size,
        'gumbel_temperature': self._gumbel_temperature,
        'seed': self._seed,
    })
    return config

  def __call__(self, y_true, y_pred, sample_weight=None):
    """See _RankingLoss."""
    # For Gumbel approx NDCG, the logits are sampled from Gumbel distribution
    # to sort the documents.
    gbl_labels, gbl_logits, gbl_weights = self._gumbel_sampler.sample(
        y_true, y_pred, weights=sample_weight)
    return super().__call__(gbl_labels, gbl_logits, gbl_weights)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ClickEMLoss(_RankingLoss):
  r"""Computes click EM loss between `y_true` and `y_pred`.

  Implementation of click EM loss ([Wang et al, 2018][wang2018]). This loss
  assumes that a click is generated by a factorized model
  $$P(\text{examination}) \cdot P(\text{relevance})$$, which are latent
  variables determined by `exam_logits` and `rel_logits` respectively.

  NOTE: This loss should be called with a `logits` tensor of shape
  `[batch_size, list_size, 2]`. The two elements in the last dimension of
  `logits` represent `exam_logits` and `rel_logits` respectively.

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[[0.6, 0.9], [0.8, 0.2]]]
  >>> loss = tfr.keras.losses.ClickEMLoss()
  >>> loss(y_true, y_pred).numpy()
  0.7981389

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[[0.6, 0.9], [0.8, 0.2]],
  ...     [[0.5, 0.9], [0.8, 0.2], [0.4, 0.8]]])
  >>> loss = tfr.keras.losses.ClickEMLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  1.0770882

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.ClickEMLoss())
  ```

  References:
    - [Position Bias Estimation for Unbiased Learning to Rank in Personal
       Search, Wang et al, 2018][wang2018].

  [wang2018]: https://research.google/pubs/pub46485/
  """

  def __init__(self,
               reduction=tf.losses.Reduction.AUTO,
               name=None,
               exam_loss_weight=1.0,
               rel_loss_weight=1.0,
               ragged=False):
    """Click EM loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      exam_loss_weight: (Optional) Weight of examination logits.
      rel_loss_weight: (Optional) Weight of relevance logits.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, ragged)
    self._exam_loss_weight = exam_loss_weight
    self._rel_loss_weight = rel_loss_weight
    self._loss = losses_impl.ClickEMLoss(
        name='{}_impl'.format(name) if name else None,
        exam_loss_weight=self._exam_loss_weight,
        rel_loss_weight=self._rel_loss_weight,
        ragged=ragged)

  def get_config(self):
    config = super().get_config()
    config.update({
        'exam_loss_weight': self._exam_loss_weight,
        'rel_loss_weight': self._rel_loss_weight,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class SigmoidCrossEntropyLoss(_RankingLoss):
  r"""Computes the Sigmoid cross-entropy loss between `y_true` and `y_pred`.

  ```
  loss = -(y_true log(sigmoid(y_pred)) + (1 - y_true) log(1 - sigmoid(y_pred)))
  ```

  NOTE: This loss does not support graded relevance labels and should only be
  used with binary relevance labels ($$y \in [0, 1]$$).

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.SigmoidCrossEntropyLoss()
  >>> loss(y_true, y_pred).numpy()
  0.8042943

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.SigmoidCrossEntropyLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.6444636

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
      loss=tfr.keras.losses.SigmoidCrossEntropyLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = - \sum_{i}
  y_i \log(\text{sigmoid}(s_i))
  + (1 - y_i) \log(1 - \text{sigmoid}(s_i))
  $$
  """

  def __init__(self, reduction=tf.losses.Reduction.AUTO, name=None,
               ragged=False):
    super().__init__(reduction, name, ragged)
    self._loss = losses_impl.SigmoidCrossEntropyLoss(
        name='{}_impl'.format(name) if name else None,
        ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class MeanSquaredLoss(_RankingLoss):
  r"""Computes mean squared loss between `y_true` and `y_pred`.

  ```
  loss = (y_true - y_pred)^2
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.MeanSquaredLoss()
  >>> loss(y_true, y_pred).numpy()
  0.4

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.MeanSquaredLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.20833336

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.MeanSquaredLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = \sum_i (y_i - s_i)^{2}
  $$
  """

  def __init__(self, reduction=tf.losses.Reduction.AUTO, name=None,
               ragged=False):
    """Mean squared loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, ragged)
    self._loss = losses_impl.MeanSquaredLoss(
        name='{}_impl'.format(name) if name else None, ragged=ragged)
