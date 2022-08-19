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

"""Utils for tfr.keras."""

from typing import Callable

import tensorflow as tf

TensorLike = tf.types.experimental.TensorLike
GainFunction = Callable[[TensorLike], tf.Tensor]
RankDiscountFunction = Callable[[TensorLike], tf.Tensor]
PositiveFunction = Callable[[TensorLike], tf.Tensor]


# The following functions are used to transform labels and ranks for losses and
# metrics computation. User customized functions can be defined similarly by
# following the same annotations.
@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def identity(label: TensorLike) -> tf.Tensor:
  """Identity function that returns the input label.

  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    The input label.
  """
  return label


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def inverse(rank: TensorLike) -> tf.Tensor:
  """Computes the inverse of input rank.

  Args:
    rank: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `1/x`.
  """
  return tf.math.divide_no_nan(1., rank)


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def pow_minus_1(label: TensorLike) -> tf.Tensor:
  """Computes `2**x - 1` element-wise for each label.

  Can be used to define `gain_fn` for `tfr.keras.metrics.NDCGMetric`.

  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `2**x - 1`.
  """
  return tf.math.pow(2., label) - 1.


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def log2_inverse(rank: TensorLike) -> tf.Tensor:
  """Computes `1./log2(1+x)` element-wise for each label.

  Can be used to define `rank_discount_fn` for `tfr.keras.metrics.NDCGMetric`.

  Args:
    rank: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `1./log2(1+x)`.
  """
  return tf.math.divide_no_nan(tf.math.log(2.), tf.math.log1p(rank))


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def is_greater_equal_1(label: TensorLike) -> tf.Tensor:
  """Computes whether label is greater or equal to 1.

  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `I(x > 1)`.
  """
  return tf.greater_equal(label, 1.0)


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def symmetric_log1p(t: TensorLike) -> tf.Tensor:
  """Computes `sign(x) * log(1 + sign(x))`.

  Args:
    t: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.

  Returns:
    A `Tensor` that has each input element transformed as `x` to `I(x > 1)`.
  """
  return tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
