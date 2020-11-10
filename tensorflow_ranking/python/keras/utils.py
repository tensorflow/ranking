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
"""Utils for tfr.keras."""

import tensorflow.compat.v2 as tf


# The following functions are used to transform labels and ranks for losses and
# metrics computation. User customized functions can be defined similarly by
# following the same annotations.
@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def identity(label):
  return label


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def inverse(rank):
  return tf.math.divide_no_nan(1., rank)


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def pow_minus_1(label):
  return tf.math.pow(2., label) - 1.


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def log2_inverse(rank):
  return tf.math.divide_no_nan(tf.math.log(2.), tf.math.log1p(rank))


@tf.keras.utils.register_keras_serializable(package="tensorflow_ranking")
def is_greater_equal_1(label):
  return tf.greater_equal(label, 1.0)
