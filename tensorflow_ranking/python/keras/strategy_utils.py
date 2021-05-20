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

"""tf.distribute strategy utils for Ranking pipeline in tfr.keras.

In TF2, the distributed training can be easily handled with Strategy offered in
tf.distribute. Depending on device and MapReduce technique, there are four
strategies are currently supported. They are:
MirroredStrategy: synchronous strategy on a single CPU/GPU worker.
MultiWorkerMirroredStrategy: synchronous strategy on multiple CPU/GPU workers.
TPUStrategy: distributed strategy working on TPU.
ParameterServerStrategy: asynchronous distributed strategy on CPU/GPU workers.

Note: ParameterServerStrategy is not fully compatible with `model.fit` in
current version of tensorflow, thus not supported.

Please check https://www.tensorflow.org/guide/distributed_training for more
information.
"""

import os
from typing import Any, Optional, Union

import tensorflow as tf

TPU_STRATEGY = "TPUStrategy"
PS_STRATEGY = "ParameterServerStrategy"
MIRRORED_STRATEGY = "MirroredStrategy"
MWMS_STRATEGY = "MultiWorkerMirroredStrategy"


def get_strategy(
    strategy: str,
    tpu: Optional[str] = ""
) -> Union[None, tf.distribute.MirroredStrategy,
           tf.distribute.MultiWorkerMirroredStrategy,
           tf.distribute.experimental.TPUStrategy,]:
  """Creates and initializes the requested tf.distribute strategy.

  Example usage:

  ```python
  strategy = get_strategy("MirroredStrategy")
  ```

  Args:
    strategy: Key for a `tf.distribute` strategy to be used to train the model.
      Choose from ["MirroredStrategy", "MultiWorkerMirroredStrategy",
      "TPUStrategy"]. If None, no distributed strategy will be used.
    tpu: TPU address for TPUStrategy. Not used for other strategy.

  Returns:
    A strategy will be used for distributed training.

  Raises:
    ValueError if `strategy` is not supported.
  """
  if strategy is None:
    return None
  elif strategy == MIRRORED_STRATEGY:
    return tf.distribute.MirroredStrategy()
  elif strategy == MWMS_STRATEGY:
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    return tf.distribute.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
  elif strategy == TPU_STRATEGY:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy
  else:
    # TODO: integrate PSStrategy to pipeline.
    raise ValueError("Unsupported strategy {}".format(strategy))


class NullContextManager(object):
  """A null context manager for local training.

  Example usage:

  ```python
  with NullContextManager():
    model = ...
  ```
  """

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


def strategy_scope(strategy: Optional[tf.distribute.Strategy]) -> Any:
  """Gets the strategy.scope() for training with strategy.

  Example usage:

  ```python
  with strategy_scope(strategy):
    model = ...
  ```

  Args:
    strategy: Distributed training strategy is used.

  Returns:
    ContextManager for the distributed training strategy.
  """
  if strategy is None:
    return NullContextManager()

  return strategy.scope()


def get_output_filepath(filepath: str,
                        strategy: Optional[tf.distribute.Strategy]) -> str:
  """Gets filepaths for different workers to resolve conflict of MWMS.

  Example usage:

  ```python
  strategy = get_strategy("MultiWorkerMirroredStrategy")
  worker_filepath = get_output_filepath("model/", strategy)
  ```

  Args:
    filepath: Path to output model files.
    strategy: Distributed training strategy is used.

  Returns:
    Output path that is compatible with strategy and the specific worker.
  """
  if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
    task_type, task_id = (strategy.cluster_resolver.task_type,
                          strategy.cluster_resolver.task_id)
    if task_type is not None and task_type != "chief":
      basepath = "workertemp_" + str(task_id) if task_id is not None else ""
      temp_dir = os.path.join(filepath, basepath)
      tf.io.gfile.makedirs(temp_dir)
      return temp_dir
  return filepath
