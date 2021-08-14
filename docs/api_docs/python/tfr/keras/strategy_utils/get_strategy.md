description: Creates and initializes the requested tf.distribute strategy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.strategy_utils.get_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.strategy_utils.get_strategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/strategy_utils.py#L29-L71">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates and initializes the requested tf.distribute strategy.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.strategy_utils.get_strategy(
    strategy: str,
    tpu: Optional[str] = &#x27;&#x27;
) -> Union[None, tf.distribute.MirroredStrategy, tf.distribute.
    MultiWorkerMirroredStrategy, tf.distribute.experimental.TPUStrategy]
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
"TPUStrategy"]. If None, no distributed strategy will be used.
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
