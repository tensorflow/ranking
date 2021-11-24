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

r"""Generates api_docs for tensorflow_ranking.

```shell
pip install git+http://github.com/tensorflow/docs
python build_docs.py --output_dir=/tmp/tfr_api
```
"""

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_ranking as tfr

# Hide these from the documentation. Nobody should be accessing things through
# `tfr.python.*` and `tfr.extension.*`
del tfr.python
del tfr.extension

# `losses_impl` and `metrics_impl` are not available under tfr namespace, see
# `dir(tfr)` for available APIs. These must be removed from the documentation.
del tfr.losses_impl
del tfr.losses.losses_impl
del tfr.metrics_impl
del tfr.metrics.metrics_impl

# Removing references to `estimator` or feature columns APIs in docstrings.
del tfr.estimator
del tfr.ext
del tfr.feature
del tfr.head
del tfr.keras.estimator
del tfr.keras.feature
del tfr.keras.network
del tfr.keras.canned
del tfr.losses  # Keras losses available via tfr.keras.losses.
del tfr.metrics  # Keras metrics available via tfr.keras.metrics.
del tfr.model


FLAGS = flags.FLAGS


flags.DEFINE_string('output_dir', '/tmp/tfr_api',
                    'Where to write the resulting docs to.')
flags.DEFINE_string(
    'code_url_prefix',
    ('https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking'),
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files.')

flags.DEFINE_string('site_path', 'ranking/api_docs/python',
                    'Path prefix in the _toc.yaml')


def build_docs(output_dir):
  """Build api docs for TensorFlow Ranking package."""
  doc_generator = generate_lib.DocGenerator(
      root_title='TensorFlow Ranking',
      py_modules=[('tfr', tfr)],
      base_dir=os.path.dirname(tfr.__file__),
      search_hints=FLAGS.search_hints,
      code_url_prefix=FLAGS.code_url_prefix,
      site_path=FLAGS.site_path,
      callbacks=[public_api.local_definitions_filter])

  doc_generator.build(output_dir)
  print('Output docs to: ', FLAGS.output_dir)


def main(_):
  build_docs(FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
