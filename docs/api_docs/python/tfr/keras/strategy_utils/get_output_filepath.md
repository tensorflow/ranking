description: Gets filepaths for different workers to resolve conflict of MWMS.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.strategy_utils.get_output_filepath" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.strategy_utils.get_output_filepath

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/strategy_utils.py#L114-L140">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Gets filepaths for different workers to resolve conflict of MWMS.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.strategy_utils.get_output_filepath(
    filepath: str,
    strategy: Optional[tf.distribute.Strategy]
) -> str
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example usage:

```python
strategy = get_strategy("MultiWorkerMirroredStrategy")
worker_filepath = get_output_filepath("model/", strategy)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filepath`
</td>
<td>
Path to output model files.
</td>
</tr><tr>
<td>
`strategy`
</td>
<td>
Distributed training strategy is used.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Output path that is compatible with strategy and the specific worker.
</td>
</tr>

</table>
