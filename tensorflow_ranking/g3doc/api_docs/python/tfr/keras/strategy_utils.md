description: tf.distribute strategy utils for Ranking pipeline in tfr.keras.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.strategy_utils" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="MIRRORED_STRATEGY"/>
<meta itemprop="property" content="MWMS_STRATEGY"/>
<meta itemprop="property" content="PS_STRATEGY"/>
<meta itemprop="property" content="TPU_STRATEGY"/>
</div>

# Module: tfr.keras.strategy_utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/strategy_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

tf.distribute strategy utils for Ranking pipeline in tfr.keras.

In TF2, the distributed training can be easily handled with Strategy offered in
tf.distribute. Depending on device and MapReduce technique, there are four
strategies are currently supported. They are: MirroredStrategy: synchronous
strategy on a single CPU/GPU worker. MultiWorkerMirroredStrategy: synchronous
strategy on multiple CPU/GPU workers. TPUStrategy: distributed strategy working
on TPU. ParameterServerStrategy: asynchronous distributed strategy on CPU/GPU
workers.

Note: ParameterServerStrategy is not fully compatible with `model.fit` in
current version of tensorflow, thus not supported.

Please check https://www.tensorflow.org/guide/distributed_training for more
information.

## Classes

[`class NullContextManager`](../../tfr/keras/strategy_utils/NullContextManager.md):
A null context manager for local training.

## Functions

[`get_output_filepath(...)`](../../tfr/keras/strategy_utils/get_output_filepath.md):
Gets filepaths for different workers to resolve conflict of MWMS.

[`get_strategy(...)`](../../tfr/keras/strategy_utils/get_strategy.md): Creates
and initializes the requested tf.distribute strategy.

[`strategy_scope(...)`](../../tfr/keras/strategy_utils/strategy_scope.md): Gets
the strategy.scope() for training with strategy.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
MIRRORED_STRATEGY<a id="MIRRORED_STRATEGY"></a>
</td>
<td>
`'MirroredStrategy'`
</td>
</tr><tr>
<td>
MWMS_STRATEGY<a id="MWMS_STRATEGY"></a>
</td>
<td>
`'MultiWorkerMirroredStrategy'`
</td>
</tr><tr>
<td>
PS_STRATEGY<a id="PS_STRATEGY"></a>
</td>
<td>
`'ParameterServerStrategy'`
</td>
</tr><tr>
<td>
TPU_STRATEGY<a id="TPU_STRATEGY"></a>
</td>
<td>
`'TPUStrategy'`
</td>
</tr>
</table>
