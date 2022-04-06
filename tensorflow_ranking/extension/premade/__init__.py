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

"""TensorFlow Ranking Premade Orbit Task Module.

Note: First - These APIs require These APS require the
`tensorflow_models`package. You can install it with `pip install
tf-models-official`. Second - Nothing under
`tensorflow_ranking.extension.premade` is imported by default. To use
these APIs import `premade` in your code:
`import tensorflow_ranking.extension.premade`.
"""

from tensorflow_ranking.extension.premade.tfrbert_task import *  # pylint: disable=wildcard-import,line-too-long
