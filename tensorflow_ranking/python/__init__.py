# Copyright 2019 The TensorFlow Ranking Authors.
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

"""TensorFlow Ranking library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_ranking.python import data
from tensorflow_ranking.python import feature
from tensorflow_ranking.python import head
from tensorflow_ranking.python import losses
from tensorflow_ranking.python import metrics
from tensorflow_ranking.python import model
from tensorflow_ranking.python import utils
from tensorflow_ranking.python import multi_head

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-bad-import-order

_allowed_symbols = [
    'data', 'feature', 'head', 'losses', 'metrics', 'model', 'utils', 'multi_head'
]

remove_undocumented(__name__, _allowed_symbols)
