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

"""Tests for utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_ranking.python import utils


class UtilsTest(tf.test.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    tf.compat.v1.reset_default_graph()

  def test_is_label_valid(self):
    with tf.Graph().as_default():
      labels = [[1.0, 0.0, -1.0]]
      labels_validity = [[True, True, False]]
      with tf.compat.v1.Session() as sess:
        is_valid = sess.run(utils.is_label_valid(labels))
        self.assertAllEqual(is_valid, labels_validity)

  def test_sort_by_scores(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      positions = [[1, 2, 3], [4, 5, 6]]
      names = [['a', 'b', 'c'], ['d', 'e', 'f']]
      with tf.compat.v1.Session() as sess:
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

  def test_sort_by_scores_for_3d_features(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.], [1., 2., 3.]]
      example_feature = [[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                         [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]]]
      with tf.compat.v1.Session() as sess:
        sorted_example_feature = sess.run(
            utils.sort_by_scores(scores, [example_feature])[0])
        self.assertAllEqual(
            sorted_example_feature,
            [[[4., 5., 6.], [7., 8., 9.], [1., 2., 3.]],
             [[70., 80., 90.], [40., 50., 60.], [10., 20., 30.]]])

        sorted_example_feature = sess.run(
            utils.sort_by_scores(scores, [example_feature], topn=2)[0])
        self.assertAllEqual(
            sorted_example_feature,
            [[[4., 5., 6.], [7., 8., 9.]], [[70., 80., 90.], [40., 50., 60.]]])

        sorted_example_feature = sess.run(
            utils.sort_by_scores([scores[0]], [[example_feature[0]]])[0])
        self.assertAllEqual(sorted_example_feature,
                            [[[4., 5., 6.], [7., 8., 9.], [1., 2., 3.]]])

  def test_sort_by_scores_shuffle_ties(self):
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(589)
      scores = [[2., 1., 1.]]
      names = [['a', 'b', 'c']]
      with tf.compat.v1.Session() as sess:
        sorted_names = sess.run(
            utils.sort_by_scores(scores, [names], shuffle_ties=False))[0]
        self.assertAllEqual(sorted_names, [[b'a', b'b', b'c']])
        sorted_names = sess.run(
            utils.sort_by_scores(scores, [names], shuffle_ties=True, seed=2))[0]
        self.assertAllEqual(sorted_names, [[b'a', b'c', b'b']])

  def test_sorted_ranks(self):
    with tf.Graph().as_default():
      scores = [[1., 3., 2.]]
      with tf.compat.v1.Session() as sess:
        ranks = sess.run(utils.sorted_ranks(scores, seed=1))
        self.assertAllEqual(ranks, [[3, 1, 2]])

      tf.compat.v1.set_random_seed(3)
      scores = [[1., 2., 1.]]
      with tf.compat.v1.Session() as sess:
        ranks = sess.run(utils.sorted_ranks(scores, shuffle_ties=False, seed=1))
        self.assertAllEqual(ranks, [[2, 1, 3]])
        ranks = sess.run(utils.sorted_ranks(scores, shuffle_ties=True, seed=1))
        self.assertAllEqual(ranks, [[3, 1, 2]])

  def test_organize_valid_indices(self):
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(4)
      labels = [[1.0, 0.0, -1.0], [-1.0, 1.0, 2.0]]
      is_valid = utils.is_label_valid(labels)
      shuffled_indices = utils.shuffle_valid_indices(is_valid, seed=2)
      organized_indices = utils.organize_valid_indices(is_valid, shuffle=False)
      with tf.compat.v1.Session() as sess:
        shuffled_indices = sess.run(shuffled_indices)
        self.assertAllEqual(
            shuffled_indices,
            [[[0, 1], [0, 0], [0, 2]], [[1, 1], [1, 2], [1, 0]]])
        organized_indices = sess.run(organized_indices)
        self.assertAllEqual(
            organized_indices,
            [[[0, 0], [0, 1], [0, 2]], [[1, 1], [1, 2], [1, 0]]])

  def test_reshape_first_ndims_dense_tensor(self):
    with tf.Graph().as_default():
      # Batch size = 2, list size = 5, embedding size = 10.
      tensor = tf.reshape(tf.range(100), shape=(2, 5, 10))
      target_tensor = tf.reshape(tf.range(100), shape=(10, 10))
      reshaped_tensor = utils.reshape_first_ndims(tensor, 2, [10])
      self.assertAllEqual(reshaped_tensor.get_shape().as_list(), [10, 10])

      with tf.compat.v1.Session() as sess:
        reshaped, target = sess.run([reshaped_tensor, target_tensor])
        self.assertAllEqual(reshaped, target)

  def test_reshape_first_ndims_sparse_tensor(self):
    with tf.Graph().as_default():
      # Batch size = 2, list size = 3, embedding size = 3.
      # Tensor:
      # [[[1, 0, 0], [0, 2, 0], [0, 0, 3]], [[4, 0, 0], [0, 5, 0], [0, 0, 6]]].
      # Reshaped :
      # [[[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 0, 0], [0, 5, 0], [0, 0, 6]]].
      sparse_tensor = tf.SparseTensor(
          indices=[[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 0], [1, 1, 1],
                   [1, 2, 2]],
          values=[1, 2, 3, 4, 5, 6],
          dense_shape=[2, 3, 3])
      target = tf.SparseTensor(
          indices=[[0, 0], [1, 1], [2, 2], [3, 0], [4, 1], [5, 2]],
          values=[1, 2, 3, 4, 5, 6],
          dense_shape=[6, 3])
      reshaped = utils.reshape_first_ndims(sparse_tensor, 2, [6])
      with tf.compat.v1.Session() as sess:
        reshaped_array, target_array = sess.run([reshaped, target])
        self.assertAllEqual(reshaped_array.indices, target_array.indices)
        self.assertAllEqual(reshaped_array.values, target_array.values)
        self.assertAllEqual(reshaped_array.dense_shape,
                            target_array.dense_shape)

  def test_reshape_to_2d(self):
    with tf.Graph().as_default():
      tensor_3d = tf.constant([[[1], [2], [3]], [[4], [5], [6]]])
      tensor_3d_reshaped = utils.reshape_to_2d(tensor_3d)
      tensor_1d = tf.constant([1, 2, 3])
      tensor_1d_reshaped = utils.reshape_to_2d(tensor_1d)
      with tf.compat.v1.Session() as sess:
        self.assertAllEqual(
            sess.run(tensor_3d_reshaped), [[1, 2, 3], [4, 5, 6]])
        self.assertAllEqual(sess.run(tensor_1d_reshaped), [[1], [2], [3]])

  def test_circular_indices(self):
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        # All valid.
        indices, mask = sess.run(utils._circular_indices(3, [3]))
        self.assertAllEqual(indices, [[0, 1, 2]])
        self.assertAllEqual(mask, [[True, True, True]])
        # One invalid.
        indices, mask = sess.run(utils._circular_indices(3, [2]))
        self.assertAllEqual(indices, [[0, 1, 0]])
        self.assertAllEqual(mask, [[True, True, False]])
        # All invalid.
        indices, mask = sess.run(utils._circular_indices(3, [0]))
        self.assertAllEqual(indices, [[0, 0, 0]])
        self.assertAllEqual(mask, [[False, False, False]])
        # batch_size = 2
        indices, mask = sess.run(utils._circular_indices(3, [3, 2]))
        self.assertAllEqual(indices, [[0, 1, 2], [0, 1, 0]])
        self.assertAllEqual(mask, [[True, True, True], [True, True, False]])

  def test_padded_nd_indices(self):
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(1)
      # batch_size, list_size = 2, 3.
      is_valid = [[True, True, True], [True, True, False]]
      # Disable shuffling.
      indices, mask = utils.padded_nd_indices(is_valid, shuffle=False)

      with tf.compat.v1.Session() as sess:
        indices, mask = sess.run([indices, mask])
        # shape = [2, 3, 2] = [batch_size, list_size, 2].
        self.assertAllEqual(
            indices,
            [  # batch_size = 2.
                [  # list_size = 3.
                    [0, 0], [0, 1], [0, 2]
                ],
                [  # list_size = 3.
                    [1, 0], [1, 1], [1, 0]
                ]
            ])
        # shape = [2, 3] = [batch_size, list_size]
        self.assertAllEqual(mask, [[True, True, True], [True, True, False]])

      # Enable shuffling.
      indices, mask = utils.padded_nd_indices(
          is_valid, shuffle=True, seed=87124)

      with tf.compat.v1.Session() as sess:
        indices, mask = sess.run([indices, mask])
        # shape = [2, 3, 2] = [batch_size, list_size, 2].
        self.assertAllEqual(
            indices,
            [  # batch_size = 2.
                [  # list_size = 3.
                    [0, 0], [0, 1], [0, 2]
                ],
                [  # list_size = 3
                    [1, 1], [1, 0], [1, 1]
                ]
            ])
        # shape = [2, 3] = [batch_size, list_size]
        self.assertAllEqual(mask, [[True, True, True], [True, True, False]])

  def test_de_noise(self):
    with tf.Graph().as_default():
      counts = [[1, 2, 3], [1, 2, 3]]
      noise = [[3, 3, 4], [3, 2, 1]]
      with tf.compat.v1.Session() as sess:
        # Larger noise ratio -> the results are more sharp.
        de_noised = sess.run(utils.de_noise(counts, noise, ratio=0.9))
        self.assertAllClose(de_noised, [[0., 0.22, 0.78], [0., 0., 1.]])
        # Smaller noise ratio -> the results are close to counts.
        de_noised = sess.run(utils.de_noise(counts, noise, ratio=0.1))
        self.assertAllClose(de_noised, [
            [0.151852, 0.337037, 0.511111],
            [0.12963, 0.333333, 0.537037],
        ])

  def test_de_noise_exception(self):
    with tf.Graph().as_default():
      with self.assertRaises(ValueError):
        utils.de_noise([[1, 2, 3]], [[1, 2, 3]], ratio=1.1)
      with self.assertRaises(ValueError):
        utils.de_noise([[1, 2, 3]], [[1, 2, 3]], ratio=-0.1)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        utils.de_noise([[-1, 2, 3]], [[1, 2, 3]])
      with self.assertRaises(tf.errors.InvalidArgumentError):
        utils.de_noise([[1, 2, 3]], [[0, 2, 3]])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
