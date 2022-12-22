# Copyright 2022 The TensorFlow Ranking Authors.
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

"""Keras losses in TF-Ranking."""

from typing import Any, Dict, List, Optional

import tensorflow.compat.v2 as tf

from tensorflow_ranking.python import losses_impl
from tensorflow_ranking.python.keras import utils


class RankingLossKey(object):
  """Ranking loss key strings."""
  # Names for the ranking based loss functions.
  PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
  PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
  PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
  PAIRWISE_MSE_LOSS = 'pairwise_mse_loss'
  SOFTMAX_LOSS = 'softmax_loss'
  UNIQUE_SOFTMAX_LOSS = 'unique_softmax_loss'
  SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
  MEAN_SQUARED_LOSS = 'mean_squared_loss'
  ORDINAL_LOSS = 'ordinal_loss'
  LIST_MLE_LOSS = 'list_mle_loss'
  APPROX_NDCG_LOSS = 'approx_ndcg_loss'
  APPROX_MRR_LOSS = 'approx_mrr_loss'
  GUMBEL_APPROX_NDCG_LOSS = 'gumbel_approx_ndcg_loss'
  COUPLED_RANKDISTIL_LOSS = 'coupled_rankdistil_loss'
  # TODO: Add support for circle loss and neural sort losses.

  @classmethod
  def all_keys(cls) -> List[str]:
    return [v for k, v in vars(cls).items() if k.isupper()]


def get(loss: str,
        reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
        lambda_weight: Optional[losses_impl._LambdaWeight] = None,
        name: Optional[str] = None,
        **kwargs) -> tf.keras.losses.Loss:
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
      RankingLossKey.ORDINAL_LOSS: OrdinalLoss,
      RankingLossKey.APPROX_NDCG_LOSS: ApproxNDCGLoss,
      RankingLossKey.APPROX_MRR_LOSS: ApproxMRRLoss,
      RankingLossKey.GUMBEL_APPROX_NDCG_LOSS: GumbelApproxNDCGLoss,
      RankingLossKey.COUPLED_RANKDISTIL_LOSS: CoupledRankDistilLoss,
  }

  key_to_cls_with_lambda_weights = {
      RankingLossKey.LIST_MLE_LOSS: ListMLELoss,
      RankingLossKey.PAIRWISE_HINGE_LOSS: PairwiseHingeLoss,
      RankingLossKey.PAIRWISE_LOGISTIC_LOSS: PairwiseLogisticLoss,
      RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS: PairwiseSoftZeroOneLoss,
      RankingLossKey.PAIRWISE_MSE_LOSS: PairwiseMSELoss,
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
class LabelDiffLambdaWeight(losses_impl.LabelDiffLambdaWeight):
  """Keras serializable class for LabelDiffLambdaWeight."""

  def __init__(self, **kwargs):
    super().__init__()

  def get_config(self) -> Dict[str, Any]:
    return {}


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class DCGLambdaWeight(losses_impl.DCGLambdaWeight):
  """Keras serializable class for DCG."""

  def __init__(self,
               topn: Optional[int] = None,
               gain_fn: Optional[utils.GainFunction] = None,
               rank_discount_fn: Optional[utils.RankDiscountFunction] = None,
               normalized: bool = False,
               smooth_fraction: float = 0.,
               **kwargs):
    gain_fn = gain_fn or utils.identity
    rank_discount_fn = rank_discount_fn or utils.inverse
    super().__init__(topn, gain_fn, rank_discount_fn, normalized,
                     smooth_fraction)

  def get_config(self) -> Dict[str, Any]:
    return {
        'topn': self._topn,
        'gain_fn': self._gain_fn,
        'rank_discount_fn': self._rank_discount_fn,
        'normalized': self._normalized,
        'smooth_fraction': self._smooth_fraction,
    }


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class NDCGLambdaWeightV2(losses_impl.DCGLambdaWeightV2):
  """Keras serializable class for NDCG LambdaWeight V2 for topn."""

  def __init__(self,
               topn: Optional[int] = None,
               gain_fn: Optional[utils.GainFunction] = None,
               rank_discount_fn: Optional[utils.RankDiscountFunction] = None,
               **kwargs):
    gain_fn = gain_fn or utils.pow_minus_1
    rank_discount_fn = rank_discount_fn or utils.log2_inverse
    super().__init__(topn, gain_fn, rank_discount_fn, normalized=True)

  def get_config(self) -> Dict[str, Any]:
    return {
        'topn': self._topn,
        'gain_fn': self._gain_fn,
        'rank_discount_fn': self._rank_discount_fn,
    }


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class NDCGLambdaWeight(DCGLambdaWeight):
  """Keras serializable class for NDCG."""

  def __init__(self,
               topn: Optional[int] = None,
               gain_fn: Optional[utils.GainFunction] = None,
               rank_discount_fn: Optional[utils.RankDiscountFunction] = None,
               smooth_fraction: float = 0.,
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

  def __init__(self,
               topn: Optional[int] = None,
               positive_fn: Optional[utils.PositiveFunction] = None,
               **kwargs):
    positive_fn = positive_fn or utils.is_greater_equal_1
    super().__init__(topn, positive_fn)

  def get_config(self) -> Dict[str, Any]:
    return {
        'topn': self._topn,
        'positive_fn': self._positive_fn,
    }


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class ListMLELambdaWeight(losses_impl.ListMLELambdaWeight):

  def __init__(self,
               rank_discount_fn: Optional[utils.RankDiscountFunction] = None,
               **kwargs):
    super().__init__(rank_discount_fn)

  def get_config(self) -> Dict[str, Any]:
    return {
        'rank_discount_fn': self._rank_discount_fn,
    }


class _RankingLoss(tf.keras.losses.Loss):
  """Base class for all ranking losses.

  Please see tf.keras.losses.Loss for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               ragged: bool = False):
    super().__init__(reduction, name)
    # An instance of loss in `losses_impl`. Overwrite this in subclasses.
    self._loss = None
    self._ragged = ragged

  def __call__(self,
               y_true: utils.TensorLike,
               y_pred: utils.TensorLike,
               sample_weight: Optional[utils.TensorLike] = None) -> tf.Tensor:
    """See tf.keras.losses.Loss."""
    if self._loss is None:
      raise ValueError('self._loss is not defined. Please use a subclass.')
    sample_weight = self._loss.normalize_weights(y_true, sample_weight)
    return super().__call__(y_true, y_pred, sample_weight)

  def call(self, y_true: utils.TensorLike,
           y_pred: utils.TensorLike) -> tf.Tensor:
    """See tf.keras.losses.Loss."""
    y_pred = self._loss.get_logits(y_pred)
    losses, weights = self._loss.compute_unreduced_loss(
        labels=y_true, logits=y_pred)
    return tf.multiply(losses, weights)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({'ragged': self._ragged})
    return config


class _PairwiseLoss(_RankingLoss):
  """Base class for pairwise ranking losses."""

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False,
               **kwargs):
    super().__init__(reduction, name, ragged)
    self._lambda_weight = lambda_weight
    self._temperature = temperature

  def get_config(self) -> Dict[str, Any]:
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

  def call(self, y_true: utils.TensorLike,
           y_pred: utils.TensorLike) -> tf.Tensor:
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
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
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
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
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
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
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


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class PairwiseMSELoss(_PairwiseLoss):
  r"""Computes pairwise mean squared error loss between `y_true` and `y_pred`.

  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:

  ```
  loss = sum_{i \neq j} ((s_i - s_j) - (y_i - y_j))**2
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[0.6, 0.8]]
  >>> loss = tfr.keras.losses.PairwiseMSELoss()
  >>> loss(y_true, y_pred).numpy()
  1.44

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[0.6, 0.8], [0.5, 0.8, 0.4]])
  >>> loss = tfr.keras.losses.PairwiseMSELoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.7666667

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tfr.keras.losses.PairwiseMSELoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) =
  \sum_{i \neq j}((s_i - s_j) - (y_i - y_j))^2
  $$
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
    """Pairwise Mean Squared Error loss.

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
    self._loss = losses_impl.PairwiseMSELoss(
        name='{}_impl'.format(name) if name else None,
        lambda_weight=lambda_weight,
        temperature=temperature,
        ragged=ragged)


class _ListwiseLoss(_RankingLoss):
  """Base class for listwise ranking losses."""

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False,
               **kwargs):
    super().__init__(reduction, name, ragged)
    self._lambda_weight = lambda_weight
    self._temperature = temperature

  def get_config(self) -> Dict[str, Any]:
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
  \mathcal{L}(\{y\}, \{s\}) = - \sum_i y_i
  \log\left(\frac{\exp(s_i)}{\sum_j \exp(s_j)}\right)
  $$
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
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

  def __call__(self,
               y_true: utils.TensorLike,
               y_pred: utils.TensorLike,
               sample_weight: Optional[utils.TensorLike] = None) -> tf.Tensor:
    """See _RankingLoss."""
    losses, sample_weight = self._loss.compute_per_list(y_true, y_pred,
                                                        sample_weight)
    return tf.keras.__internal__.losses.compute_weighted_loss(
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
  \mathcal{L}(\{y\}, \{s\}) = - \sum_i (2^{y_i} - 1)
  \log\left(\frac{\exp(s_i)}{\sum_j I_{y_i > y_j} \exp(s_j) + \exp(s_i)}\right)
  $$

  References:
    - [Listwise Learning to Rank by Exploring Unique Ratings, Zhu et al,
       2020][zhu2020]

  [zhu2020]: https://arxiv.org/abs/2001.01828
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
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

  where $P(\pi_y | s)$ is the Plackett-Luce probability of a permutation
  $\pi_y$ conditioned on scores $s$. Here $\pi_y$ represents a permutation
  of items ordered by the relevance labels $y$ where ties are broken randomly.

  References:
    - [Listwise approach to learning to rank: theory and algorithm, Xia et al,
       2008][xia2008]

  [xia2008]: https://dl.acm.org/doi/10.1145/1390156.1390306
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               ragged: bool = False):
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

  [qin2008]: https://dl.acm.org/doi/10.1007/s10791-009-9124-x
  """  # pylint: disable=g-line-too-long

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 0.1,
               ragged: bool = False):
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
  -0.655107

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

  [qin2008]: https://dl.acm.org/doi/10.1007/s10791-009-9124-x
  [bruch2019]: https://research.google/pubs/pub48168/
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 0.1,
               ragged: bool = False):
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
  -0.6987189

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
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 0.1,
               sample_size: int = 8,
               gumbel_temperature: float = 1.0,
               seed: Optional[int] = None,
               ragged: bool = False):
    super().__init__(
        reduction, name, lambda_weight, temperature=temperature, ragged=ragged)
    self._sample_size = sample_size
    self._gumbel_temperature = gumbel_temperature
    self._seed = seed

    self._gumbel_sampler = losses_impl.GumbelSampler(
        name=name,
        sample_size=sample_size,
        temperature=gumbel_temperature,
        seed=seed,
        ragged=ragged)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        'sample_size': self._sample_size,
        'gumbel_temperature': self._gumbel_temperature,
        'seed': self._seed,
    })
    return config

  def __call__(self,
               y_true: utils.TensorLike,
               y_pred: utils.TensorLike,
               sample_weight: Optional[utils.TensorLike] = None) -> tf.Tensor:
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
  $P(\text{examination}) \cdot P(\text{relevance})$, which are latent
  variables determined by `exam_logits` and `rel_logits` respectively.

  NOTE: This loss should be called with a `logits` tensor of shape
  `[batch_size, list_size, 2]`. The two elements in the last dimension of
  `logits` represent `exam_logits` and `rel_logits` respectively.

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[[0.6, 0.9], [0.8, 0.2]]]
  >>> loss = tfr.keras.losses.ClickEMLoss()
  >>> loss(y_true, y_pred).numpy()
  1.1462884

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
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               exam_loss_weight: float = 1.0,
               rel_loss_weight: float = 1.0,
               ragged: bool = False):
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

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        'exam_loss_weight': self._exam_loss_weight,
        'rel_loss_weight': self._rel_loss_weight,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class MixtureEMLoss(_ListwiseLoss):
  r"""Computes mixture EM loss between `y_true` and `y_pred`.

  Implementation of mixture Expectation-Maximization loss
  ([Yan et al, 2018][yan2022]). This loss assumes that the clicks in a session
  are generated by one of mixture models.

  NOTE: This loss should be called with a `logits` tensor of shape
  `[batch_size, list_size, model_num]`. The elements in the last dimension of
  `logits` represent models to be mixed.

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[[0.6, 0.9], [0.8, 0.2]]]
  >>> loss = tfr.keras.losses.MixtureEMLoss()
  >>> loss(y_true, y_pred).numpy()
  1.3198698

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[1., 0.], [0., 1., 0.]])
  >>> y_pred = tf.ragged.constant([[[0.6, 0.9], [0.8, 0.2]],
  ...     [[0.5, 0.9], [0.8, 0.2], [0.4, 0.8]]])
  >>> loss = tfr.keras.losses.MixtureEMLoss(ragged=True)
  >>> loss(y_true, y_pred).numpy()
  1.909512

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tfr.keras.losses.MixtureEMLoss())
  ```

  References:
    - [Revisiting two tower models for unbiased learning to rank, Yan et al,
       2022][yan2022].

  [yan2022]: https://research.google/pubs/pub51296/
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               lambda_weight: Optional[losses_impl._LambdaWeight] = None,
               temperature: float = 1.0,
               alpha: float = 1.0,
               ragged: bool = False):
    """Mixture EM loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      lambda_weight: (Optional) A lambdaweight to apply to the loss. Can be one
        of `tfr.keras.losses.DCGLambdaWeight`,
        `tfr.keras.losses.NDCGLambdaWeight`, or,
        `tfr.keras.losses.PrecisionLambdaWeight`.
      temperature: (Optional) The temperature to use for scaling the logits.
      alpha: (Optional) The smooth factor of the probability.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
    """
    super().__init__(reduction, name, lambda_weight, temperature, ragged)
    self._alpha = alpha
    self._loss = losses_impl.MixtureEMLoss(
        name='{}_impl'.format(name) if name else None,
        alpha=self._alpha,
        ragged=ragged)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        'alpha': self._alpha,
    })
    return config


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class SigmoidCrossEntropyLoss(_RankingLoss):
  r"""Computes the Sigmoid cross-entropy loss between `y_true` and `y_pred`.

  ```
  loss = -(y_true log(sigmoid(y_pred)) + (1 - y_true) log(1 - sigmoid(y_pred)))
  ```

  NOTE: This loss does not support graded relevance labels and should only be
  used with binary relevance labels ($y \in [0, 1]$).

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
  0.64446354

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
      loss=tfr.keras.losses.SigmoidCrossEntropyLoss())
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = - \sum_{i} y_i
  \log(\text{sigmoid}(s_i)) + (1 - y_i) \log(1 - \text{sigmoid}(s_i))
  $$
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               ragged: bool = False):
    super().__init__(reduction, name, ragged)
    self._loss = losses_impl.SigmoidCrossEntropyLoss(
        name='{}_impl'.format(name) if name else None, ragged=ragged)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class MeanSquaredLoss(_RankingLoss):
  r"""Computes mean squared loss between `y_true` and `y_pred`.

  ```
  loss = (y_true - y_pred)**2
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

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               ragged: bool = False):
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


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class OrdinalLoss(_RankingLoss):
  r"""Computes the Ordinal loss between `y_true` and `y_pred`.

  In ordinal loss, y_pred is a 3D tensor with the last dimension equals to
  ordinal_size.
  ```
  loss = -\sum_i=0^ordinal_size-1 I_{y_true > i} log(sigmoid(y_pred[i])) +
      I_{y_true <= i} log(1-sigmoid(y_pred[i]))
  ```

  Standalone usage:

  >>> y_true = [[1., 0.]]
  >>> y_pred = [[[0.6, 0.2], [0.8, 0.3]]]
  >>> loss = tfr.keras.losses.OrdinalLoss(ordinal_size=2)
  >>> loss(y_true, y_pred).numpy()
  1.6305413

  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[2., 1.], [0.]])
  >>> y_pred = tf.ragged.constant([[[0.6, 0.2], [0.8, 0.3]], [[0., -0.2]]])
  >>> loss = tfr.keras.losses.OrdinalLoss(ordinal_size=2, ragged=True)
  >>> loss(y_true, y_pred).numpy()
  0.88809216

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tfr.keras.losses.OrdinalLoss(ordinal_size=2))
  ```

  Definition:

  $$
  \mathcal{L}(\{y\}, \{s\}) = - \sum_i\sum_{j=0}^{m-1} I_{y_i > j}
      \log(\text{sigmoid}(s_{i,j})) + I_{y_i \leq j}
      \log(1 - \text{sigmoid}(s_{i,j}))
  $$
  """

  def __init__(self,
               reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
               name: Optional[str] = None,
               ragged: bool = False,
               ordinal_size: int = 1,
               use_fraction_label: bool = False):
    super().__init__(reduction, name, ragged)
    self._loss = losses_impl.OrdinalLoss(
        name='{}_impl'.format(name) if name else None,
        ordinal_size=ordinal_size,
        ragged=ragged,
        use_fraction_label=use_fraction_label)


@tf.keras.utils.register_keras_serializable(package='tensorflow_ranking')
class CoupledRankDistilLoss(_RankingLoss):
  r"""Computes the Rank Distil loss between `y_true` and `y_pred`.

  The Coupled-RankDistil loss ([Reddi et al, 2021][reddi2021]) is the
  cross-entropy between k-Plackett's probability of logits (student) and labels
  (teacher).

  Standalone usage:

  >>> tf.random.set_seed(1)
  >>> y_true = [[0., 2., 1.], [1., 0., 2.]]
  >>> ln = tf.math.log
  >>> y_pred = [[0., ln(3.), ln(2.)], [0., ln(2.), ln(3.)]]
  >>> loss = tfr.keras.losses.CoupledRankDistilLoss(topk=2, sample_size=1)
  >>> loss(y_true, y_pred).numpy()
  2.138333

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tfr.keras.losses.CoupledRankDistilLoss())
  ```

  Definition:

  The k-Plackett's probability model is defined as:
  $$
  \mathcal{P}_k(\pi|s) = \frac{1}{(N-k)!} \\
  \frac{\prod_{i=1}^k exp(s_{\pi(i)})}{\sum_{j=k}^N log(exp(s_{\pi(i)}))}.
  $$

  The Coupled-RankDistil loss is defined as:
  $$
  \mathcal{L}(y, s) = -\sum_{\pi} \mathcal{P}_k(\pi|y) log\mathcal{P}(\pi|s) \\
  =  \mathcal{E}_{\pi \sim \matcal{P}(.|y)} [-\log \mathcal{P}(\pi|s)]
  $$

  References:
    - [RankDistil: Knowledge Distillation for Ranking, Reddi et al,
       2021][reddi2021]

  [reddi2021]: https://research.google/pubs/pub50695/
  """

  def __init__(
      self,
      reduction: tf.losses.Reduction = tf.losses.Reduction.AUTO,
      name: Optional[str] = None,
      ragged: bool = False,
      sample_size: int = 8,
      topk: Optional[int] = None,
      temperature: Optional[float] = 1.,
  ):
    """Coupled Rank Distil loss.

    Args:
      reduction: (Optional) The `tf.keras.losses.Reduction` to use (see
        `tf.keras.losses.Loss`).
      name: (Optional) The name for the op.
      ragged: (Optional) If True, this loss will accept ragged tensors. If
        False, this loss will accept dense tensors.
      sample_size: (Optional) Number of permutations to sample from teacher
        scores. Defaults to 8.
      topk: (Optional) top-k entries over which order is matched. A penalty is
        applied over non top-k items. Defaults to `None`, which treats top-k as
        all entries in the list.
      temperature: (Optional) A float number to modify the logits as
        `logits=logits/temperature`. Defaults to 1.
    """
    super().__init__(reduction, name, ragged)
    self._sample_size = sample_size
    self._topk = topk
    self._temperature = temperature
    self._loss = losses_impl.CoupledRankDistilLoss(
        name='{}_impl'.format(name) if name else None,
        temperature=temperature,
        topk=topk,
        sample_size=sample_size,
        ragged=ragged)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        'sample_size': self._sample_size,
        'topk': self._topk,
        'temperature': self._temperature,
    })
    return config
