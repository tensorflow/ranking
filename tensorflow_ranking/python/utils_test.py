# Copyright 2018 The TensorFlow Ranking Authors.
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

"""Tests for utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from tensorflow_ranking.python import utils


class UtilsTest(test.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    ops.reset_default_graph()

  def test_is_label_valid(self):
    labels = [[1.0, 0.0, -1.0]]
    labels_validity = [[True, True, False]]
    with session.Session() as sess:
      is_valid = sess.run(utils.is_label_valid(labels))
      self.assertAllEqual(is_valid, labels_validity)

  def test_sort_by_scores(self):
    scores = [[1., 3., 2.], [1., 2., 3.]]
    positions = [[1, 2, 3], [4, 5, 6]]
    names = [['a', 'b', 'c'], ['d', 'e', 'f']]
    with session.Session() as sess:
      sorted_positions, sorted_names = sess.run(
          utils.sort_by_scores(scores, [positions, names]))
      self.assertAllEqual(sorted_positions, [[2, 3, 1], [6, 5, 4]])
      self.assertAllEqual(sorted_names,
                          [[b'b', b'c', b'a'], [b'f', b'e', b'd']])

      sorted_positions, sorted_names = sess.run(
          utils.sort_by_scores(scores, [positions, names], topn=2))
      self.assertAllEqual(sorted_positions, [[2, 3], [6, 5]])
      self.assertAllEqual(sorted_names, [[b'b', b'c'], [b'f', b'e']])

      sorted_positions, sorted_names = sess.run(
          utils.sort_by_scores([scores[0]], [[positions[0]], [names[0]]]))
      self.assertAllEqual(sorted_positions, [[2, 3, 1]])
      self.assertAllEqual(sorted_names, [[b'b', b'c', b'a']])

  def test_shuffle_valid_indices(self):
    random_seed.set_random_seed(1)
    labels = [[1.0, 0.0, -1.0], [-1.0, 1.0, 2.0]]
    is_valid = utils.is_label_valid(labels)
    shuffled_indices = utils.shuffle_valid_indices(is_valid)
    with session.Session() as sess:
      shuffled_indices = sess.run(shuffled_indices)
      self.assertAllEqual(shuffled_indices,
                          [[[0, 1], [0, 0], [0, 2]], [[1, 1], [1, 2], [1, 0]]])

  def test_reshape_first_ndims_dense_tensor(self):
    # Batch size = 2, list size = 5, embedding size = 10.
    tensor = array_ops.reshape(math_ops.range(100), shape=(2, 5, 10))
    target_tensor = array_ops.reshape(math_ops.range(100), shape=(10, 10))
    reshaped_tensor = utils.reshape_first_ndims(tensor, 2, [10])
    self.assertAllEqual(reshaped_tensor.get_shape().as_list(), [10, 10])

    with session.Session() as sess:
      reshaped, target = sess.run([reshaped_tensor, target_tensor])
      self.assertAllEqual(reshaped, target)

  def test_reshape_first_ndims_sparse_tensor(self):
    # Batch size = 2, list size = 3, embedding size = 3.
    # Tensor:
    # [[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[4, 0, 0], [0, 5, 0], [0, 0, 6]]].
    # Reshaped :
    # [[[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 0, 0], [0, 5, 0], [0, 0, 6]]].
    sparse_tensor = sparse_tensor_lib.SparseTensor(
        indices=[[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 0], [1, 1, 1],
                 [1, 2, 2]],
        values=[1, 2, 3, 4, 5, 6],
        dense_shape=[2, 3, 3])
    target = sparse_tensor_lib.SparseTensor(
        indices=[[0, 0], [1, 1], [2, 2], [3, 0], [4, 1], [5, 2]],
        values=[1, 2, 3, 4, 5, 6],
        dense_shape=[6, 3])
    reshaped = utils.reshape_first_ndims(sparse_tensor, 2, [6])
    with session.Session() as sess:
      reshaped_array, target_array = sess.run([reshaped, target])
      self.assertAllEqual(reshaped_array.indices, target_array.indices)
      self.assertAllEqual(reshaped_array.values, target_array.values)
      self.assertAllEqual(reshaped_array.dense_shape, target_array.dense_shape)


if __name__ == '__main__':
  test.main()
