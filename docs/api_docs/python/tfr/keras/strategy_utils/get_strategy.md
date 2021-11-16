description: Creates and initializes the requested tf.distribute strategy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.strategy_utils.get_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.strategy_utils.get_strategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/strategy_utils.py#L31-L102">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates and initializes the requested tf.distribute strategy.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.strategy_utils.get_strategy(
    strategy: str,
    cluster_resolver: Optional[tf.distribute.cluster_resolver.ClusterResolver] = None,
    variable_partitioner: Optional[tf.distribute.experimental.partitioners.Partitioner] = _USE_DEFAULT_VARIABLE_PARTITIONER,
    tpu: Optional[str] = &#x27;&#x27;
) -> Union[None, tf.distribute.MirroredStrategy, tf.distribute.
    MultiWorkerMirroredStrategy, tf.distribute.experimental.
    ParameterServerStrategy, tf.distribute.experimental.TPUStrategy]
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example usage:

```python
strategy = get_strategy("MirroredStrategy")
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`strategy`
</td>
<td>
Key for a `tf.distribute` strategy to be used to train the model.
Choose from ["MirroredStrategy", "MultiWorkerMirroredStrategy",
"ParameterServerStrategy", "TPUStrategy"]. If None, no distributed
strategy will be used.
</td>
</tr><tr>
<td>
`cluster_resolver`
</td>
<td>
A cluster_resolver to build strategy.
</td>
</tr><tr>
<td>
`variable_partitioner`
</td>
<td>
Variable partitioner to be used in
ParameterServerStrategy. If the argument is not specified, a recommended
`tf.distribute.experimental.partitioners.MinSizePartitioner` is used. If
the argument is explicitly specified as `None`, no partitioner is used and
that variables are not partitioned. This arg is used only when the
strategy is `tf.distribute.experimental.ParameterServerStrategy`.
See `tf.distribute.experimental.ParameterServerStrategy` class doc for
more information.
</td>
</tr><tr>
<td>
`tpu`
</td>
<td>
TPU address for TPUStrategy. Not used for other strategy.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A strategy will be used for distributed training.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if `strategy` is not supported.
</td>
</tr>

</table>
