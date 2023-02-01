description: Parses the encoded key to keys and weights.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfr.utils.parse_keys_and_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfr.utils.parse_keys_and_weights

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/python/utils.py#L432-L461">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Parses the encoded key to keys and weights.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfr.utils.parse_keys_and_weights(
    key: str
) -> Dict[str, float]
</code></pre>

<!-- Placeholder for "Used in" -->

This parse function will remove all spaces. Different keys are split by "," and
then weight associated with key is split by ":".

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A string represents a key, or a string of multiple keys, split by ",",
and weighted by the weights split by ":". For example, key =
'softmax_loss:0.9,sigmoid_cross_entropy_loss:0.1'.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict from keys to weights.
</td>
</tr>

</table>
