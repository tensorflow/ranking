description: Gets the strategy.scope() for training with strategy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.keras.strategy_utils.strategy_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.keras.strategy_utils.strategy_scope

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/keras/strategy_utils.py#L92-L111">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Gets the strategy.scope() for training with strategy.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.keras.strategy_utils.strategy_scope(
    strategy: Optional[tf.distribute.Strategy]
) -> Any
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example usage:

```python
with strategy_scope(strategy):
  model = ...
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
ContextManager for the distributed training strategy.
</td>
</tr>

</table>
